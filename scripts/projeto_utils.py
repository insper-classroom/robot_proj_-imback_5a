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
from sklearn.ensemble import RandomForestRegressor

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


def girar(giro, w):
    delta_t = giro/w
    vel = Twist(Vector3(0,0,0), Vector3(0,0,w))
    rospy.sleep(delta_t)
    


def escolhe_mascara_regressao(mask,bgr):
    try :    
        contornos = encontrar_contornos(mask)
        cv2.drawContours(mask, contornos, -1, [0, 0, 255], 2)

        mask_bgr = center_of_mass_region(mask, 20, 400, bgr.shape[1] - 80, bgr.shape[0]-100)

        
        img, X, Y = encontrar_centro_dos_contornos(mask_bgr, contornos)

        img = desenhar_linha_entre_pontos(mask_bgr, X,Y, (255,0,0))

        # Regressão Linear
        
        # Regressão pelo centro por Regressao Linear 
        img, lm = regressao_por_centro(img, X,Y)


        angulo = angulo_com_vertical(img, lm)

        
        str_angulo = "Angulo=%4.0f "%(angulo)

        cv2.putText(img, str_angulo, (0, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)

        return angulo, img

    
    except:
                
        return 90,bgr

    