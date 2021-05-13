from __future__ import print_function, division

import rospy 
import numpy as np
import cv2
import tf
from tf import transformations
from tf import TransformerROS
import tf2_ros
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Vector3, Pose, Vector3Stamped
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R
import visao_module
import math
import projeto2_utils as putils

#roslaunch my_simulation forca.launch

id = 0

#-- Font for the text in the image
font = cv2.FONT_HERSHEY_PLAIN
ranges = None
minv = 0
maxv = 10

bridge = CvBridge()

cv_image = None
media = []
centro = []
atraso = 1.5E9 # 1 segundo e meio. Em nanossegundos

frame = "camera_link"
# frame = "head_camera"  # DESCOMENTE para usar com webcam USB via roslaunch tag_tracking usbcam



def quart_to_euler(orientacao):
    """
    Converter quart. para euler (XYZ)
    Retorna apenas o Yaw (wz)
    """
    r = R.from_quat(orientacao)
    wx, wy, wz = (r.as_euler('xyz', degrees=True))

    return wz

x_odom = -1000
y_odom = -1000


def recebeu_leitura(dado):
    """
        Grava nas variáveis x,y,z a posição extraída da odometria
        Atenção: *não coincidem* com o x,y,z locais do drone
    """
    global x_odom
    global y_odom
    x_odom = dado.pose.pose.position.x
    y_odom = dado.pose.pose.position.y
    

## ROS
def mypose(msg):
    """
    Recebe a Leitura da Odometria.
    Para esta aplicacao, apenas a orientacao esta sendo usada
    """   
    x = msg.pose.pose.orientation.x
    y = msg.pose.pose.orientation.y
    z = msg.pose.pose.orientation.z
    w = msg.pose.pose.orientation.w

    orientacao_robo = [[x,y,z,w]]


def scaneou(dado):
    """
    Rebe a Leitura do Lidar
    Para esta aplicacao, apenas a menor distancia esta sendo usada
    """
    global distancia
    
    ranges = np.array(dado.ranges).round(decimals=2)
    distancia = ranges[0]



## Variáveis novas criadas pelo gabarito

centro_yellow = (320,240)
frame = 0
skip = 3
m = 0
angle_yellow = 0 # angulo com a vertical

low = putils.low
high = putils.high

centro_caixa = (320, 240)


## 
distance = 0

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


# A função a seguir é chamada sempre que chega um novo frame
def roda_todo_frame(imagem):
    global centro_yellow
    global m
    global angle_yellow
    global cv_image
    global resultados
    global ids
    global distance
    global distancenp
    global state
    global x_odom
    global y_odom
    global state
       

    try:
        cv_image = bridge.compressed_imgmsg_to_cv2(imagem, "bgr8")
        #cv2.imshow("Camera", cv_image)
        ##
        copia = cv_image.copy() # se precisar usar no while

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        


        if frame%skip==0: # contamos a cada skip frames

            mask = putils.filter_color(copia, low, high)          

            img, centro_yellow  =  putils.center_of_mass_region(mask, 0, 300, mask.shape[1], mask.shape[0])  

            saida_bgr, m, h = putils.ajuste_linear_grafico_x_fy(mask)

            ang = math.atan(m)
            ang_deg = math.degrees(ang)

            angle_yellow = ang_deg

            putils.texto(saida_bgr, f"Angulo graus: {ang_deg}", (15,50), color=(0,255,255))
            putils.texto(saida_bgr, f"Angulo rad: {ang}", (15,90), color=(0,255,255))

            cv2.imshow("centro", img)
            cv2.imshow("angulo", saida_bgr)

            putils.aruco_reader(cv_image,ids,corners,marker_size,camera_matrix,camera_distortion,font)
            str_ids = f"ID: {ids}"
            cv2.putText(cv_image, str_ids, (0, 50), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

            str_odom = "x = %5.4f y = %5.4f"%(x_odom, y_odom)
            
            cv2.putText(cv_image, str_odom, (0, 130), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

            ## Achando o maior objeto azul 
            #media, centro_frame, area = putils.identifica_cor(copia)
            
            #area_caixa = area
            #centro_caixa = media
        


        cv2.imshow("cv_image", cv_image)
        
        cv2.waitKey(1)
    except CvBridgeError as e:
        print('ex', e)





if __name__=="__main__":

    rospy.init_node("q3")

    topico_imagem = "/camera/image/compressed"
    velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size = 3 )
    cmd_vel = velocidade_saida

    recebe_scan = rospy.Subscriber("/scan", LaserScan, scaneou)
    pose_sub = rospy.Subscriber('/odom', Odometry , mypose)
    recebe_scan = rospy.Subscriber('/odom', Odometry , recebeu_leitura)
    recebedor = rospy.Subscriber(topico_imagem, CompressedImage, roda_todo_frame, queue_size=4, buff_size = 2**24)

    zero = Twist(Vector3(0,0,0), Vector3(0,0,0))         


    x = 0
    
    tol_centro = 10 # tolerancia de fuga do centro
    tol_ang = 15 # tolerancia do angulo


    c_img = (320,240) # Centro da imagem  que ao todo é 640 x 480

    v_slow = 0.3
    v_rapido = 0.85
    w_slow = 0.2
    w_rapido = 0.75

    
    INICIAL= -1
    AVANCA = 0
    AVANCA_RAPIDO = 1
    ALINHA = 2
    AVANCA_PROXIMO = 3
    TERMINOU = 4
    BIFURCAR = 5 
    BIFURCAR_ESQUERDA = 6

    state = INICIAL

    def inicial():
        # Ainda sem uma ação específica
        pass

    def avanca():
        vel = Twist(Vector3(v_slow,0,0), Vector3(0,0,0)) 
        cmd_vel.publish(vel) 

    def avanca_rapido():
        vel = Twist(Vector3(v_slow,0,0), Vector3(0,0,0))         
        cmd_vel.publish(vel)

    def alinha():
        delta_x = c_img[x] - centro_yellow[x]
        max_delta = 150.0
        w = (delta_x/max_delta)*w_rapido
        vel = Twist(Vector3(v_slow,0,0), Vector3(0,0,w)) 
        cmd_vel.publish(vel)        

    def avanca_proximo():
       pass

    def terminou():
        zero = Twist(Vector3(0,0,0), Vector3(0,0,0))         
        cmd_vel.publish(zero)

    def bifurcar():
        vel = Twist(Vector3(0.2,0,0), Vector3(0,0,0.5)) 
        cmd_vel.publish(vel)

    def bifurcar_esquerda():
        w = 0.2
        giro = math.radians(45)
        delta_t = giro/0.2
        vel = Twist(Vector3(1,0,0), Vector3(0,0,w))
        cmd_vel.publish(vel)
        rospy.sleep(delta_t)
        
        
        


    def dispatch():
        "Logica de determinar o proximo estado"
        global state
        global ids
        global distance
        global x_odom
        global y_odom


        if c_img[x] - tol_centro < centro_yellow[x] < c_img[x] + tol_centro:
            state = AVANCA
            if   - tol_ang< angle_yellow  < tol_ang:  # para angulos centrados na vertical, regressao de x = f(y) como está feito
                state = AVANCA_RAPIDO

            if  x_odom < -2.19 and y_odom > -0.15:# and ids is not None:
                zero = Twist(Vector3(0,0,0), Vector3(0,0,0))         
                cmd_vel.publish(zero)
                #state = BIFURCAR_ESQUERDA
                w = 15
                giro = math.radians(40)
                delta_t = giro/w
                vel = Twist(Vector3(1,0,0), Vector3(0,0,w))
                cmd_vel.publish(vel)
                rospy.sleep(delta_t)
                state = AVANCA

            #     # if distance < 100:

            # if x_odom < -2.5:
                


            # if  -2.60 > x_odom > -2.73 and -0.28 > y_odom > -0.37:
            #     zero = Twist(Vector3(0,0,0), Vector3(0,0,0))         
            #     cmd_vel.publish(zero)
            #     #state = BIFURCAR_ESQUERDA
            #     w = 15
            #     giro = math.radians(40)
            #     delta_t = giro/w
            #     vel = Twist(Vector3(1,0,0), Vector3(0,0,w))
            #     cmd_vel.publish(vel)
            #     rospy.sleep(delta_t)
            #     state = AVANCA
            
        else: 
                state = ALINHA

        print("centro_yellow {} angle_yellow {:.3f} state: {}".format(centro_yellow, angle_yellow, state))
        

    acoes = {INICIAL:inicial, AVANCA: avanca, AVANCA_RAPIDO: avanca_rapido, ALINHA: alinha, AVANCA_PROXIMO: avanca_proximo , TERMINOU: terminou, BIFURCAR: bifurcar, BIFURCAR_ESQUERDA: bifurcar_esquerda}


    r = rospy.Rate(200) 

    while not rospy.is_shutdown():
        print("Estado: ", state)       
        acoes[state]()  # executa a funcão que está no dicionário
        dispatch()            
        r.sleep()