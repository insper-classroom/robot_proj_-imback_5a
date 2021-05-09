#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import print_function, division
import rospy
import numpy as np
import numpy
import tf
import math
import cv2
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from numpy import linalg
from tf import transformations
from tf import TransformerROS
import tf2_ros
from geometry_msgs.msg import Twist, Vector3, Pose, Vector3Stamped
import cv2.aruco as aruco
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from sklearn.linear_model import LinearRegression
import visao_module
import projeto_utils as utils 

#print("EXECUTE ANTES da 1.a vez: ")
#print("wget https://github.com/Insper/robot21.1/raw/main/projeto/ros_projeto/scripts/MobileNetSSD_deploy.caffemodel")
#print("PARA TER OS PESOS DA REDE NEURAL")

bridge = CvBridge()

cv_image = None
media = []
centro = []
atraso = 1.5E9 # 1 segundo e meio. Em nanossegundos


area = 0.0 # Variavel com a area do maior contorno

# Só usar se os relógios ROS da Raspberry e do Linux desktop estiverem sincronizados. 
# Descarta imagens que chegam atrasadas demais
check_delay = False 

resultados = [] # Criacao de uma variavel global para guardar os resultados vistos

x = 0
y = 0
z = 0 
id = 0

frame = "camera_link"
# frame = "head_camera"  # DESCOMENTE para usar com webcam USB via roslaunch tag_tracking usbcam

#-- Font for the text in the image
font = cv2.FONT_HERSHEY_PLAIN

tfl = 0

tf_buffer = tf2_ros.Buffer()

ponto_fuga = (320, 240)   

angulo = 90

vel = Twist(Vector3(0,0,0), Vector3(0,0,0.2))

SEGUIR = True
BIFURCAR = False
VOLTAR = False



def image_callback(img_cv):

    global angulo 
    global ids
    global distancenp
    global vel 
    global distance
    global SEGUIR
    global BIFURCAR
    global VOLTAR

    # BEGIN BRIDGE
    #image = bridge.imgmsg_to_cv2(msg)
    # END BRIDGE
    # BEGIN HSV
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    # END HSV
    # BEGIN FILTER

    lower_yellow = numpy.array([25, 50, 50])
    upper_yellow = numpy.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    kernel = np.ones((5,5),np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask_copy_esquerda = mask.copy()
    mask_copy_direita = mask.copy()

    mask_esquerda  = mask_copy_esquerda[:,0:320]
    mask_direita = mask_copy_direita[:,320:640]

    #cv2.imshow("Mascara Esquerda", mask_esquerda)
    #cv2.imshow("Mascara Direita", mask_direita)


    # END FILTER
    masked = cv2.bitwise_and(img_cv, img_cv, mask=mask)
    #cv2.imshow("Filtra Amarelo", mask ) 
    cv2.waitKey(3)

    #bgr = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)\
    bgr = img_cv.copy()

    contornos = utils.encontrar_contornos(mask)
    cv2.drawContours(mask, contornos, -1, [0, 0, 255], 2)

    mask_bgr = utils.center_of_mass_region(mask, 20, 400, bgr.shape[1] - 80, bgr.shape[0]-100)

    
    img, X, Y = utils.encontrar_centro_dos_contornos(mask_bgr, contornos)

    img = utils.desenhar_linha_entre_pontos(mask_bgr, X,Y, (255,0,0))

    # Regressão Linear
    
    ## Regressão pelo centro
    img, lm = utils.regressao_por_centro(img, X,Y)


    angulo = utils.angulo_com_vertical(img, lm)
    print(f' Angulo com a vertical{angulo}')

    #zero = Twist(Vector3(0,0,0), Vector3(0,0,0))
    #esq = Twist(Vector3(0.1,0,0), Vector3(0,0,0.2))
    #dire = Twist(Vector3(0.1,0,0), Vector3(0,0,-0.2))    
    #frente = Twist(Vector3(0.4,0,0), Vector3(0,0,0))  

    if SEGUIR:

        if distance > 100:

            if angulo is None:
                vel = Twist(Vector3(0,0,0), Vector3(0,0,0))

            else:

                if angulo > 90:
                    if angulo < 150:
                        vel = Twist(Vector3(0.1,0,0), Vector3(0,0,-0.2))
                    else:
                        vel = Twist(Vector3(0.4,0,0), Vector3(0,0,0)) 
                else:
                    if angulo > 30:
                        vel = Twist(Vector3(0.1,0,0), Vector3(0,0,0.2))
                    else:
                        vel = Twist(Vector3(0.4,0,0), Vector3(0,0,0))  

        else:
            vel = Twist(Vector3(0,0,0), Vector3(0,0,0))
            SEGUIR,BIFURCAR = False, True


    if BIFURCAR:

        utils.girar(velocidade_saida, giro, -0.4)

        if distance  > 10:

            if angulo is None:
                vel = Twist(Vector3(0,0,0), Vector3(0,0,0))

            else:

                if angulo > 90:
                    if angulo < 150:
                        vel = Twist(Vector3(0.1,0,0), Vector3(0,0,-0.2))
                    else:
                        vel = Twist(Vector3(0.4,0,0), Vector3(0,0,0)) 
                else:
                    if angulo > 30:
                        vel = Twist(Vector3(0.1,0,0), Vector3(0,0,0.2))
                    else:
                        vel = Twist(Vector3(0.4,0,0), Vector3(0,0,0))

        else:
            vel = Twist(Vector3(0,0,0), Vector3(0,0,0))
            BIFURCAR,VOLTAR = False, True


    if VOLTAR:
        
        utils.girar(velocidade_saida, np.radians(360), -0.4)

        if angulo is None:
                vel = Twist(Vector3(0,0,0), Vector3(0,0,0))

        else:

            if angulo > 90:
                if angulo < 150:
                    vel = Twist(Vector3(0.1,0,0), Vector3(0,0,-0.2))
                else:
                    vel = Twist(Vector3(0.4,0,0), Vector3(0,0,0)) 
            else:
                if angulo > 30:
                    vel = Twist(Vector3(0.1,0,0), Vector3(0,0,0.2))
                else:
                    vel = Twist(Vector3(0.4,0,0), Vector3(0,0,0))


    cv2.imshow("Regressao", img)
   

    cv2.waitKey(3)


def scaneou(dado):
	#print("scan")
	global scan_dist 
	scan_dist = dado.ranges[0]*100
	return scan_dist


ids = 0 
id_to_find  = 100
marker_size  = 25 
#--- Get the camera calibration path
calib_path  = "/home/borg/catkin_ws/src/robot202/ros/exemplos202/scripts/"
camera_matrix   = np.loadtxt(calib_path+'cameraMatrix_raspi.txt', delimiter=',')
camera_distortion   = np.loadtxt(calib_path+'cameraDistortion_raspi.txt', delimiter=',')

aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters  = aruco.DetectorParameters_create()
parameters.minDistanceToBorder = 0

scan_dist = 0

distance = 151
distancenp = 0

# A função a seguir é chamada sempre que chega um novo frame
def roda_todo_frame(imagem):
    print("frame")
    global cv_image
    global media
    global centro
    global resultados
    global ids
    global distance
    global distancenp

    now = rospy.get_rostime()
    imgtime = imagem.header.stamp
    lag = now-imgtime # calcula o lag
    delay = lag.nsecs
    # print("delay ", "{:.3f}".format(delay/1.0E9))
    if delay > atraso and check_delay==True:
        # Esta logica do delay so' precisa ser usada com robo real e rede wifi 
        # serve para descartar imagens antigas
        print("Descartando por causa do delay do frame:", delay)
        return 
    try:
        temp_image = bridge.compressed_imgmsg_to_cv2(imagem, "bgr8")
        # Note que os resultados já são guardados automaticamente na variável
        # chamada resultados
        centro, saida_net, resultados =  visao_module.processa(temp_image)        
        for r in resultados:
            # print(r) - print feito para documentar e entender
            # o resultado            
            pass

        # Desnecessário - Hough e MobileNet já abrem janelas
        cv_image = saida_net.copy()
        saida_amarelo = image_callback(cv_image)

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        print(ids)

        if ids is not None:
            #-- ret = [rvec, tvec, ?]
            #-- rvec = [[rvec_1], [rvec_2], ...] vetor de rotação
            #-- tvec = [[tvec_1], [tvec_2], ...] vetor de translação
            ret = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)
            rvec, tvec = ret[0][0,0,:], ret[1][0,0,:]

            #-- Desenha um retanculo e exibe Id do marker encontrado
            aruco.drawDetectedMarkers(cv_image, corners, ids) 
            aruco.drawAxis(cv_image, camera_matrix, camera_distortion, rvec, tvec, 1)

            #-- Print tvec vetor de tanslação em x y z
            str_position = "Marker x=%4.0f  y=%4.0f  z=%4.0f"%(tvec[0], tvec[1], tvec[2])
            print(str_position)
            cv2.putText(cv_image, str_position, (0, 100), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

            ##############----- Referencia dos Eixos------###########################
            # Linha referencia em X
            #cv2.line(cv_image, (cv_image.shape[1]/2,cv_image.shape[0]/2), ((cv_image.shape[1]/2 + 50),(cv_image.shape[0]/2)), (0,0,255), 5) 
            # Linha referencia em Y
            #cv2.line(cv_image, (cv_image.shape[1]/2,cv_image.shape[0]/2), (cv_image.shape[1]/2,(cv_image.shape[0]/2 + 50)), (0,255,0), 5) 	

            #####################---- Distancia Euclidiana ----#####################
            # Calcula a distancia usando apenas a matriz tvec, matriz de tanslação
            # Pode usar qualquer uma das duas formas
            distance = np.sqrt(tvec[0]**2 + tvec[1]**2 + tvec[2]**2)
            distancenp = np.linalg.norm(tvec)

            #-- Print distance
            str_dist = "Dist aruco=%4.0f  dis.np=%4.0f"%(distance, distancenp)
            print(str_dist)
            cv2.putText(cv_image, str_dist, (0, 15), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

            #####################---- Distancia pelo foco ----#####################
            #https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/

            # raspicam v2 focal legth 
            FOCAL_LENGTH = 3.6 #3.04
            # pixel por unidade de medida
            m = (camera_matrix[0][0]/FOCAL_LENGTH + camera_matrix[1][1]/FOCAL_LENGTH)/2
            # corners[0][0][0][0] = [ID][plano?][pos_corner(sentido horario)][0=valor_pos_x, 1=valor_pos_y]	
            pixel_length1 = math.sqrt(math.pow(corners[0][0][0][0] - corners[0][0][1][0], 2) + math.pow(corners[0][0][0][1] - corners[0][0][1][1], 2))
            pixel_length2 = math.sqrt(math.pow(corners[0][0][2][0] - corners[0][0][3][0], 2) + math.pow(corners[0][0][2][1] - corners[0][0][3][1], 2))
            pixlength = (pixel_length1+pixel_length2)/2
            dist = marker_size * FOCAL_LENGTH / (pixlength/m)

            #-- Print distancia focal
            str_distfocal = "Dist focal=%4.0f"%(dist)
            print(str_distfocal)
            cv2.putText(cv_image, str_distfocal, (0, 30), font, 1, (0, 255, 0), 1, cv2.LINE_AA)	


            ####################--------- desenha o cubo -----------#########################
            # https://github.com/RaviJoshii/3DModeler/blob/eb7ca48fa06ca85fcf5c5ec9dc4b562ce9a22a76/opencv/program/detect.py			
            m = marker_size/2
            pts = np.float32([[-m,m,m], [-m,-m,m], [m,-m,m], [m,m,m],[-m,m,0], [-m,-m,0], [m,-m,0], [m,m,0]])
            imgpts, _ = cv2.projectPoints(pts, rvec, tvec, camera_matrix, camera_distortion)
            imgpts = np.int32(imgpts).reshape(-1,2)
            cv_image = cv2.drawContours(cv_image, [imgpts[:4]],-1,(0,0,255),4)
            for i,j in zip(range(4),range(4,8)): cv_image = cv2.line(cv_image, tuple(imgpts[i]), tuple(imgpts[j]),(0,0,255),4);
            cv_image = cv2.drawContours(cv_image, [imgpts[4:]],-1,(0,0,255),4)
		
        
        cv2.imshow("cv_image", cv_image)
        cv2.waitKey(1)

    except CvBridgeError as e:
        print('ex', e)
    

if __name__=="__main__":
    rospy.init_node("cor")


    topico_imagem = "/camera/image/compressed"

    recebedor = rospy.Subscriber(topico_imagem, CompressedImage, roda_todo_frame, queue_size=4, buff_size = 2**24)


    print("Usando ", topico_imagem)

    velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)

    tfl = tf2_ros.TransformListener(tf_buffer) #conversao do sistema de coordenadas 
    tolerancia = 25
   
    centro  = 320
    margem = 5

    try:
               
        
        while not rospy.is_shutdown():
        
            
            velocidade_saida.publish(vel)
            rospy.sleep(0.1)

            #for r in resultados:
                #print(r)


    except rospy.ROSInterruptException:
        print("Ocorreu uma exceção com o rospy")


