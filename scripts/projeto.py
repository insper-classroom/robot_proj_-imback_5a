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

from nav_msgs.msg import Odometry
from std_msgs.msg import Header


from sklearn.linear_model import LinearRegression

#print("EXECUTE ANTES da 1.a vez: ")
#print("wget https://github.com/Insper/robot21.1/raw/main/projeto/ros_projeto/scripts/MobileNetSSD_deploy.caffemodel")
#print("PARA TER OS PESOS DA REDE NEURAL")


import visao_module


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

tfl = 0

tf_buffer = tf2_ros.Buffer()

def encontrar_centro_dos_contornos(img, contornos):
    """Não mude ou renomeie esta função
        deve receber um contorno e retornar, respectivamente, a imagem com uma cruz no centro de cada segmento e o centro dele. formato: img, x, y
    """
    centrox = []
    centroy = []

    for i in contornos:
        centro_x2, centro_y2 = center_of_mass(i)
        centrox.append(centro_x2)
        centroy.append(centro_y2)
        crosshair(img,(centro_x2, centro_y2), 5, (255,0,0))

    return img, centrox, centroy

def encontrar_contornos(mask):
    """Não mude ou renomeie esta função
        deve receber uma imagem preta e branca os contornos encontrados
    """
    contornos, arvore = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contornos

def crosshair(img, point, size, color):
    """ Desenha um crosshair centrado no point.
        point deve ser uma tupla (x,y)
        color é uma tupla R,G,B uint8
    """
    x,y = point
    cv2.line(img,(x - size,y),(x + size,y),color,2)
    cv2.line(img,(x,y - size),(x, y + size),color,2)

def center_of_mass(mask):
    """ Retorna uma tupla (cx, cy) que desenha o centro do contorno"""
    M = cv2.moments(mask)
    # Usando a expressão do centróide definida em: https://en.wikipedia.org/wiki/Image_moment
    
    m00 = M["m00"]

    if m00 == 0:
        m00 = 1

    cX = int(M["m10"] / m00)
    cY = int(M["m01"] / m00)
    return [int(cX), int(cY)]

def center_of_mass_region(mask, x1, y1, x2, y2):
    # Para fins de desenho
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    clipped = mask[y1:y2, x1:x2]
    c = center_of_mass(clipped)
    c[0]+=x1
    c[1]+=y1
    crosshair(mask_bgr, c, 10, (0,0,255))
    cv2.rectangle(mask_bgr, (x1, y1), (x2, y2), (255,0,0),2,cv2.LINE_AA)
    return mask_bgr


def desenhar_linha_entre_pontos(img, X, Y, color):
    """Não mude ou renomeie esta função
        deve receber uma lista de coordenadas XY, e retornar uma imagem com uma linha entre os centros EM SEQUENCIA do mais proximo.
    """
    for i in range(1,len(X)):
        cv2.line(img,(X[i-1],Y[i-1]),(X[i],Y[i]),color,2)
    
    return img

def regressao_por_centro(img, x,y):
    """Não mude ou renomeie esta função
        deve receber uma lista de coordenadas XY, e estimar a melhor reta, utilizando o metodo preferir, que passa pelos centros. Retorne a imagem com a reta e os parametros da reta
        
        Dica: cv2.line(img,ponto1,ponto2,color,2) desenha uma linha que passe entre os pontos, mesmo que ponto1 e ponto2 não pertençam a imagem.
    """
    x_shape = np.array(x)
    y_shape = np.array(y)

    x_shape = x_shape.reshape(-1,1)
    y_shape = y_shape.reshape(-1,1)

    regression = LinearRegression()

    regression.fit(x_shape, y_shape)

    w, z = regression.coef_, regression.intercept_

    a_1 = 100
    a_2 = 10000
    b_1 = int((a_1*w+z))
    b_2 = int((a_2*w+z))
    ponto_a = (a_1,b_1)
    ponto_b = (a_2,b_2)

    print(ponto_a, ponto_b)

    cor = (255,0,0)

    cv2.line(img,ponto_a,ponto_b,cor,2)

    return img, (w, z)

angulo = 90

def angulo_com_vertical(img, lm):
    global angulo 
    radianos = math.atan(lm[0])
    angulo = 90 + math.degrees(radianos)
    return angulo

def intersect_segs(seg1, seg2):
    m1,h1 = find_m_h(seg1)
    m2,h2 = find_m_h(seg2)
    x_i = (h2 - h1)/(m1-m2)
    y_i = m1*x_i + h1
    return x_i, y_i

def morpho_limpa(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask = cv2.morphologyEx( mask, cv2.MORPH_OPEN, kernel )
    mask = cv2.morphologyEx( mask, cv2.MORPH_CLOSE, kernel )    
    return mask

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


ponto_fuga = (320, 240)   

'''def encontra_pf(bgr_in):
    """
       Recebe imagem bgr e retorna
       tupla (x,y) com a posicao do ponto de fuga
    """
    
    bgr = bgr_in.copy()

    print()

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower_yellow = numpy.array([25, 50, 50])
    upper_yellow = numpy.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    bordas = auto_canny(mask)

    #print("Tamanho da tela", mask.shape) 

    lines = cv2.HoughLinesP(image = bordas, rho = 1, theta = math.pi/180.0, threshold = 40, lines= np.array([]), minLineLength = 30, maxLineGap = 5)

    if lines is None:
        return

    a,b,c = lines.shape

    bordas = morpho_limpa(bordas)

    hough_img_rgb = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)

    neg = []
    pos = []

    for i in range(a):
        # Faz uma linha ligando o ponto inicial ao ponto final, com a cor vermelha (BGR)
        cv2.line(hough_img_rgb, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (80, 80, 80), 5, cv2.LINE_AA)
        x1, y1, x2, y2 = lines[i][0][0], lines[i][0][1], lines[i][0][2], lines[i][0][3]
        reta = ((x1, y1), (x2, y2))
        m = (y2 - y1)/ (x2 - x1)

        if m >= 0.1: 
            pos.append(reta) 
        elif m < -0.1:
            neg.append(reta)


    if len(neg) >=1 and len(pos)>=1:
        # Escolher algum para calcular ponto de fuga
        # Alternativas:
        # a. mais longa de cada lado
        # b. primeira
        # c. sortear
        rneg = random.choice(neg)
        rpos = random.choice(pos)

        cv2.line(hough_img_rgb, rneg[0], rneg[1], (0, 255, 0), 5, cv2.LINE_AA)
        cv2.line(hough_img_rgb, rpos[0], rpos[1], (255, 0, 0), 5, cv2.LINE_AA)

        pf = intersect_segs(rneg, rpos)

        # Tratamento apenas para caso em que intersecoes nao sao encontradas: 
        if not np.isnan(pf[0]) and not np.isnan(pf[1]) : 
            pfi = (int(pf[0]), int(pf[1]))
            crosshair(hough_img_rgb, pfi, 10, (255,255,255))
            global ponto_fuga 
            ponto_fuga = pfi

    cv2.imshow("Saida pf ", hough_img_rgb)    '''



def image_callback(img_cv):
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
    # END FILTER
    masked = cv2.bitwise_and(img_cv, img_cv, mask=mask)
    cv2.imshow("Filtra Amarelo", mask ) 
    cv2.waitKey(3)

    #bgr = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)\
    bgr = img_cv.copy()

    contornos = encontrar_contornos(mask)
    cv2.drawContours(mask, contornos, -1, [0, 0, 255], 2)

    mask_bgr = center_of_mass_region(mask, 20, 400, bgr.shape[1] - 80, bgr.shape[0]-100)

    
    img, X, Y = encontrar_centro_dos_contornos(mask_bgr, contornos)

    img = desenhar_linha_entre_pontos(mask_bgr, X,Y, (255,0,0))

    # Regressão Linear
    
    ## Regressão pelo centro
    img, lm = regressao_por_centro(img, X,Y)

    global angulo 

    angulo = angulo_com_vertical(img, lm)
    print(angulo)


    cv2.imshow("Regressao", img)
    

    cv2.waitKey(3)




# A função a seguir é chamada sempre que chega um novo frame
def roda_todo_frame(imagem):
    print("frame")
    global cv_image
    global media
    global centro
    global resultados

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
        #pf = encontra_pf(cv_image)
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

    zero = Twist(Vector3(0,0,0), Vector3(0,0,0))
    esq = Twist(Vector3(0.2,0,0), Vector3(0,0,0.2))
    dire = Twist(Vector3(0.2,0,0), Vector3(0,0,-0.2))    
    frente = Twist(Vector3(0.2,0,0), Vector3(0,0,0))  

    centro  = 320
    margem = 5


    


    try:
               
        
        while not rospy.is_shutdown():

            """if ponto_fuga[0] <  centro - margem: 
                velocidade_saida.publish(esq)
                rospy.sleep(0.1)

                #velocidade_saida.publish(zero)
                #rospy.sleep(0.1)

            elif ponto_fuga[0] >  centro + margem: 
                velocidade_saida.publish(dire)
                rospy.sleep(0.1)
                #velocidade_saida.publish(zero)
                #rospy.sleep(0.1)

            else: 
                velocidade_saida.publish(frente)
                rospy.sleep(0.1)
                #velocidade_saida.publish(zero)
                #rospy.sleep(0.1)"""

            if angulo > 90 + margem :
                velocidade_saida.publish(dire)

            elif angulo < 90 - margem:
                velocidade_saida.publish(esq)

            else :
                velocidade_saida.publish(frente)
                         




            for r in resultados:
                print(r)

            rospy.sleep(0.1)
            
            #velocidade_saida.publish(frente)
            #rospy.sleep(0.1)

    except rospy.ROSInterruptException:
        print("Ocorreu uma exceção com o rospy")


