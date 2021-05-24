#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import print_function, division

from numpy.core.fromnumeric import put

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
from std_msgs.msg import Float64
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R
import visao_module
import math
import projeto_utils as putils

# Para rodar a simulacao faca : roslaunch my_simulation forca.launch
# Para rodar o progama: rosrun ros_projeto projeto.py
# Para rodar a garra: roslaunch mybot_description mybot_control2.launch

# ------------------------------------- DEFININDO AS VARIAVEIS ----------------------------------------------------------------------------------------------------------------
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

x_odom = -1000
y_odom = -1000

## Variáveis novas criadas pelo gabarito

centro_yellow = (320,240)
frame = 0
skip = 3
m = 0
angle_yellow = 0 # angulo com a vertical

low = putils.low
high = putils.high

## 
distancia = 0
distance = 0

ids = []
id_to_find  = 100
marker_size  = 25 
#--- Get the camera calibration path
calib_path  = "/home/borg/catkin_ws/src/robot202/ros/exemplos202/scripts/"
camera_matrix   = np.loadtxt(calib_path+'cameraMatrix_raspi.txt', delimiter=',')
camera_distortion   = np.loadtxt(calib_path+'cameraDistortion_raspi.txt', delimiter=',')

aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters  = aruco.DetectorParameters_create()
parameters.minDistanceToBorder = 0

x_bifurcacao = 1000
y_bifurcacao = 1000
x_rotatoria = 1000
y_rotatoria = 1000

centro_cor = (320, 240)
area_cor = 0

bater = True

identificaCreeper  = False


goal = ('azul', 12)

cor_desejada = goal[0]
id_desejado = goal[1]

ConceitoC = False
ConceitoB = False

# -------------------------------------- FUNCOES DE POSICOES E SENSORES ----------------------------------------------------------------------------------------------------------------------------

def quart_to_euler(orientacao):
    """
    Converter quart. para euler (XYZ)
    Retorna apenas o Yaw (wz)
    """
    r = R.from_quat(orientacao)
    wx, wy, wz = (r.as_euler('xyz', degrees=True))

    return wz


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
    #distancia = ranges[0]
    min_comeco = min(ranges[0:15])
    min_fim = min(ranges[345:360])
    distancia = min([min_comeco, min_fim])


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
    global distancia 
    global ids
    global x_bifurcacao
    global y_bifurcacao
    global x_rotatoria
    global y_rotatoria
    global centro_cor
    global area_cor
    global goal
    global identificaCreeper
    global cor_desejada
    global id_desejado
    global ConceitoB
       

    try:
        cv_image = bridge.compressed_imgmsg_to_cv2(imagem, "bgr8")
        #cv2.imshow("Camera", cv_image)
        ##
        copia = cv_image.copy() # se precisar usar no while

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        #centro, saida_net, resultados =  visao_module.processa(temp_image)    
        # for r in resultados:
            # print(r) - print feito para documentar e entender
            # o resultado            
        #     pass
        


        if frame%skip==0: # contamos a cada skip frames

            mask = putils.filter_color(copia, low, high)          

            if state == CORTAR_MASK:
                mask = mask[:,0:290] #mascara para pegar somente a bifur da esquerda 
                str_CORTAR_MASK = f'CHEGOU NA BIFURCACA0'
                cv2.putText(cv_image,str_CORTAR_MASK, (350, 90), font, 1, (150, 0, 200), 1, cv2.LINE_AA)

            else: 
                pass

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
            cv2.putText(cv_image, str_ids, (0, 50), font, 1, (255,255,255), 1, cv2.LINE_AA)

            str_odom = "x = %5.4f      y = %5.4f"%(x_odom, y_odom)
            
            cv2.putText(cv_image, str_odom, (340, 150), font, 1, (255,255,255), 1, cv2.LINE_AA)

            str_distancia = f'DISTANCIA: {distancia}'

            cv2.putText(cv_image, str_distancia, (350, 50), font, 1, (255,255,255), 1, cv2.LINE_AA)

            str_estado = f'ESTADO: {state}'

            cv2.putText(cv_image,str_estado, (350, 70), font, 1, (255,255,255), 1, cv2.LINE_AA)

            str_bifurcacao = "x_bifur = %5.2f y_bifur = %5.2f"%(x_bifurcacao, y_bifurcacao)

            cv2.putText(cv_image,str_bifurcacao, (340, 110), font, 1, (255,255,255), 1, cv2.LINE_AA)

            str_rotatoria = "x_rot = %5.2f y_rot = %5.2f"%(x_rotatoria, y_rotatoria)

            cv2.putText(cv_image,str_rotatoria, (340, 130), font, 1, (255,255,255), 1, cv2.LINE_AA)

            if identificaCreeper:
                media_cor, centro_frame, area_frame = putils.identifica_cor(cv_image,cor_desejada)

                area_cor = area_frame
                centro_cor = media_cor

                if ConceitoB:
                    str_goal = f'GOAL: achar o creeper {cor_desejada} de id {id_desejado}'
                    cv2.putText(cv_image,str_goal, (0, 20), font, 1, (0,0,0), 2, cv2.LINE_AA)
                else:
                    str_goal = f'GOAL: achar o creeper {cor_desejada}'
                    cv2.putText(cv_image,str_goal, (0, 20), font, 1, (0,0,0), 2, cv2.LINE_AA)

            else:
                str_goal = f'GOAL: seguir a pista'
                cv2.putText(cv_image,str_goal, (0, 20), font, 1, (0,0,0), 2, cv2.LINE_AA)

        cv2.imshow("cv_image", cv_image)
        cv2.waitKey(1)
        
    except CvBridgeError as e:
        print('ex', e)


if __name__=="__main__":

    rospy.init_node("cor")

    topico_imagem = "/camera/image/compressed"
    velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size = 3 )
    cmd_vel = velocidade_saida

    recebe_scan = rospy.Subscriber("/scan", LaserScan, scaneou)
    pose_sub = rospy.Subscriber('/odom', Odometry , mypose)
    recebe_scan = rospy.Subscriber('/odom', Odometry , recebeu_leitura)
    recebedor = rospy.Subscriber(topico_imagem, CompressedImage, roda_todo_frame, queue_size=4, buff_size = 2**24)
    ombro = rospy.Publisher("/joint1_position_controller/command", Float64, queue_size=1)
    garra = rospy.Publisher("/joint2_position_controller/command", Float64, queue_size=1)

    zero = Twist(Vector3(0,0,0), Vector3(0,0,0))         

    x = 0
    
    tol_centro = 10 # tolerancia de fuga do centro
    tol_ang = 15 # tolerancia do angulo


    c_img = (320,240) # Centro da imagem  que ao todo é 640 x 480

    v_slow = 0.2
    v_rapido = 0.3

    w_slow = 0.17
    w_rapido = 0.3

    
    INICIAL= -1
    AVANCA = 0
    AVANCA_RAPIDO = 1
    ALINHA = 2
    TERMINOU = 3
    PARAR = 4
    VIRAR_ESQUERDA = 5
    VIRAR_DIREITA = 6
    CORTAR_MASK = 7
    VOLTAR = 8
    FAZENDO_ROTATORIA = 9
    ALINHA_COR = 10
    SAIR_ROTATORIA = 11
    PEGA_CREEPER = 12


    segunda_volta = False

    rotatoria = False
    
    state = INICIAL

    area_ideal = 1100

    margem = 0.45


    def inicial():
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

    def terminou():
        zero = Twist(Vector3(0,0,0), Vector3(0,0,0))         
        cmd_vel.publish(zero)

    def parar():
        zero = Twist(Vector3(0,0,0), Vector3(0,0,0))         
        cmd_vel.publish(zero)
        rospy.sleep(2)
        
    def cortar_mask():
        pass
        
    def virar_esquerda():
        zero = Twist(Vector3(0,0,0), Vector3(0,0,0))         
        cmd_vel.publish(zero)
        rospy.sleep(0.5)
        w = 0.6
        giro = math.radians(60)
        delta_t = giro/w
        vel = Twist(Vector3(0,0,0), Vector3(0,0,w))
        cmd_vel.publish(vel)
        rospy.sleep(delta_t)
       
       
    def virar_direita():
        w = 20
        giro = math.radians(45)
        delta_t = giro/w
        vel = Twist(Vector3(0,0,0), Vector3(0,0,-w))
        cmd_vel.publish(vel)
        rospy.sleep(delta_t)

    def voltar():
        w = 2
        giro = math.radians(145)
        delta_t = giro/w
        vel = Twist(Vector3(0,0,0), Vector3(0,0,w))
        cmd_vel.publish(vel)
        rospy.sleep(delta_t)

    def fazendo_rotatoria():
        pass


    def alinha_cor():
        delta_x = c_img[x] - centro_cor[x]
        max_delta = 150.0
        w = (delta_x/max_delta)*w_rapido
        vel = Twist(Vector3(v_slow,0,0), Vector3(0,0,w)) 
        cmd_vel.publish(vel)    

    def sair_rotatoria():
        zero = Twist(Vector3(0,0,0), Vector3(0,0,0))         
        cmd_vel.publish(zero)
        rospy.sleep(0.5)
        w = 2
        giro = math.radians(5)
        delta_t = giro/w
        vel = Twist(Vector3(0,0,0), Vector3(0,0,w))
        cmd_vel.publish(vel)
        rospy.sleep(delta_t)

    def pega_creeper():
        # ombro.publish(-1.0) ## para baixo
        # rospy.sleep(0.3)
        # garra.publish(-1.0) ## Aberto
        # rospy.sleep(0.3)
        # ombro.publish(0.0) ## para frente
        # rospy.sleep(0.3)
        # garra.publish(0.0)  ## Fechado
        # rospy.sleep(0.3)
        # ombro.publish(1.5) ## para cima
        # rospy.sleep(0.5)
        pass


    def dispatch():
        "Logica de determinar o proximo estado"
        global state
        global ids
        global distance
        global x_odom
        global y_odom
        global distancia
        global x_bifurcacao
        global y_bifurcacao
        global x_rotatoria
        global y_rotatoria
        global rotatoria
        global segunda_volta
        global angle_yellow
        global cv_image
        global area_cor
        global centro_cor
        global area_ideal 
        global bater      
        global identificaCreeper
        global margem
        global goal
        global id_desejado
        global cor_desejada
        global ConceitoB
        global ConceitoC

        if state == PEGA_CREEPER:
            pass
        
           
        if state == VIRAR_ESQUERDA:
            rospy.sleep(1.5)
            segunda_volta = False

        if state == SAIR_ROTATORIA:
            rotatoria = False
        
        if state == PARAR:
            w = 5
            giro = math.radians(130)
            delta_t = giro/w
            vel = Twist(Vector3(0,0,0), Vector3(0,0,w))
            cmd_vel.publish(vel)
            rospy.sleep(delta_t)
            
        if state == TERMINOU:
            state = VOLTAR
                            
        if c_img[x] - tol_centro < centro_yellow[x] < c_img[x] + tol_centro:
            state = AVANCA
            if   - tol_ang< angle_yellow  < tol_ang:  # para angulos centrados na vertical, regressao de x = f(y) como está feito
                state = AVANCA_RAPIDO

            if ids is not None:
                for i in ids:
    
                    if distancia < 1.33 and i[0] == 100 :
                        state = CORTAR_MASK # corta mascara e bifurca para esquerda
                        x_bifurcacao = x_odom
                        y_bifurcacao = y_odom 
                        
                    if i[0] == 150 and distancia < 0.7:
                        state = TERMINOU #chega no final na bifurcacao da esquerda e volta para pista
                        state = VOLTAR
                        segunda_volta = True                       

                    if i[0] == 50 and distancia < 0.7:
                        state = TERMINOU #chega no final na bifurcacao da direita e volta para pista
                        state = VOLTAR
                        segunda_volta = False 

                    if i[0] == 200:
                        if 1.3 > distancia > 0.75: #entra na rotatoria pela esquerda
                            x_rotatoria = x_odom
                            y_rotatoria = y_odom
                            rotatoria = True
 
        else: 
                state = ALINHA        

        if segunda_volta:
            if x_odom < x_bifurcacao and y_odom < y_bifurcacao:
                if x_odom > x_bifurcacao - margem:
                    str_fazendo_bifur = "VAI BIFURCAR NOVAMENTE"
                    print(str_fazendo_bifur)
                    cv2.putText(cv_image,str_fazendo_bifur, (350, 90), cv2.FONT_HERSHEY_PLAIN, 1, (150, 0, 200), 1, cv2.LINE_AA)
                    state = VIRAR_ESQUERDA

        if rotatoria:
            str_fazendo_rot = "FAZENDO ROTATORIA"
            print(str_fazendo_rot)
            cv2.putText(cv_image,str_fazendo_rot, (350, 90), cv2.FONT_HERSHEY_PLAIN, 1, (150, 0, 200), 1, cv2.LINE_AA)

            if angle_yellow > 20:
                if x_odom < x_rotatoria + 0.7 and y_odom < y_rotatoria - 0.5:
                    state = SAIR_ROTATORIA
                    str_sair_rot = 'SAIR DA ROTATORIA'
                    print(str_sair_rot)
                    cv2.putText(cv_image,str_sair_rot, (350, 90), cv2.FONT_HERSHEY_PLAIN, 1, (150, 0, 200), 1, cv2.LINE_AA)
                    

        if identificaCreeper:
        
            if area_cor >= area_ideal and bater:

                if c_img[x] - tol_centro < centro_cor[x] < c_img[x] + tol_centro:
                    state = AVANCA

                    if ConceitoC:
                        if area_cor > 12500: 
                            str_creeper = 'ENCONTROU O CREEPER'
                            print(str_creeper)
                            cv2.putText(cv_image,str_creeper, (0, 100), cv2.FONT_HERSHEY_PLAIN, 1, (150, 0, 200), 1, cv2.LINE_AA)
                            state = TERMINOU 
                            state = VOLTAR
                            bater = False
                            return

                    if ConceitoB:

                        if area_cor > 10000: 

                            str_creeper = 'VAI PEGAR O CREEPER'
                            print(str_creeper)
                            cv2.putText(cv_image,str_creeper, (0, 100), cv2.FONT_HERSHEY_PLAIN, 1, (150, 0, 200), 1, cv2.LINE_AA)

                            zero = Twist(Vector3(0,0,0), Vector3(0,0,0))         
                            cmd_vel.publish(zero)
                            rospy.sleep(1)

                            ombro.publish(-0.35) ## para frente
                            rospy.sleep(2)

                            garra.publish(-1.0) ## Aberto
                            rospy.sleep(0.5)

                            vel = Twist(Vector3(0.1,0,0), Vector3(0,0,0)) 
                            cmd_vel.publish(vel) 
                            rospy.sleep(2)

                            cmd_vel.publish(zero)
                            rospy.sleep(1)

                            garra.publish(0.0)  ## Fechado
                            rospy.sleep(3.5)

                            ombro.publish(0.3) ## para cima
                            rospy.sleep(2)

                            state = TERMINOU 
                            state = VOLTAR
                            bater = False
                            return

                    
                

                else: 
                    state = ALINHA_COR

        # print("centro_cor {}  area_cor {}  state: {} ".format(centro_cor, area_cor, state))

    acoes = {INICIAL:inicial, AVANCA: avanca, AVANCA_RAPIDO: avanca_rapido, 
    ALINHA: alinha, TERMINOU: terminou, PARAR: parar, VIRAR_ESQUERDA: virar_esquerda, VIRAR_DIREITA: virar_direita,
    CORTAR_MASK: cortar_mask ,VOLTAR: voltar, FAZENDO_ROTATORIA:fazendo_rotatoria, ALINHA_COR: alinha_cor, SAIR_ROTATORIA: sair_rotatoria, PEGA_CREEPER: pega_creeper}

    r = rospy.Rate(200) 

    try:

        while not rospy.is_shutdown():
            print("Estado: ", state)       
            acoes[state]()  # executa a funcão que está no dicionário
            identificaCreeper = True
            # ConceitoC = True
            ConceitoB = True
            dispatch()   
            r.sleep()

    except rospy.ROSInterruptException:
        print("Ocorreu uma exceção com o rospy")