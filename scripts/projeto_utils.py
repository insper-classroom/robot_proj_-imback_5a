#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import print_function, division

import rospy 

import numpy as np

import cv2

from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose

from scipy.spatial.transform import Rotation as R
import cv2.aruco as aruco
import math



## Código vindo de https://github.com/Insper/robot21.1/blob/main/aula03/centro_do_amarelo.py

low = np.array([22, 50, 50],dtype=np.uint8)
high = np.array([36, 255, 255],dtype=np.uint8)

def filter_color(bgr, low, high):
    """ REturns a mask within the range"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low, high)
    return mask     

# Função centro de massa baseada na aula 02  https://github.com/Insper/robot202/blob/master/aula02/aula02_Exemplos_Adicionais.ipynb
# Esta função calcula centro de massa de máscara binária 0-255 também, não só de contorno
def center_of_mass(mask):
    """ Retorna uma tupla (cx, cy) que desenha o centro do contorno"""
    M = cv2.moments(mask)
    # Usando a expressão do centróide definida em: https://en.wikipedia.org/wiki/Image_moment
    m00 = max(M["m00"],1) # para evitar dar erro quando não há contornos
    cX = int(M["m10"] / m00)
    cY = int(M["m01"] / m00)
    return [int(cX), int(cY)]

def crosshair(img, point, size, color):
    """ Desenha um crosshair centrado no point.
        point deve ser uma tupla (x,y)
        color é uma tupla R,G,B uint8
    """
    x,y = point
    cv2.line(img,(x - size,y),(x + size,y),color,5)
    cv2.line(img,(x,y - size),(x, y + size),color,5)

def center_of_mass_region(mask, x1, y1, x2, y2):
    # Para fins de desenho
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    clipped = mask[y1:y2, x1:x2]
    c = center_of_mass(clipped)
    c[0]+=x1
    c[1]+=y1
    crosshair(mask_bgr, c, 10, (0,0,255))
    cv2.rectangle(mask_bgr, (x1, y1), (x2, y2), (255,0,0),2,cv2.LINE_AA)
    centro = (int(c[0]), int(c[1]))

    scale_percent = 50 # percent of original size
    width = int(mask_bgr.shape[1] * scale_percent / 100)
    height = int(mask_bgr.shape[0] * scale_percent / 100)
    dim = (width, height)

    mask_bgr2 = cv2.resize(mask_bgr, dim)

    return mask_bgr2, centro

font = cv2.FONT_HERSHEY_SIMPLEX

def texto(img, a, p, color = (0,50,100)):
    """Escreve na img RGB dada a string a na posição definida pela tupla p"""
    cv2.putText(img, str(a), p, font,1, color ,1,cv2.LINE_AA)
    

import statsmodels.api as sm


def ajuste_linear_x_fy(mask):
    """Recebe uma imagem já limiarizada e faz um ajuste linear
        retorna coeficientes linear e angular da reta
        e equação é da forma
        x = coef_angular*y + coef_linear
    """ 
    pontos = np.where(mask==255)
    ximg = pontos[1]
    yimg = pontos[0]

    ## Caso adicionado para evitar resultados invalidos
    if len(ximg) < 10: 
        return 0,0, [[0],[0]]

    yimg_c = sm.add_constant(yimg)
    model = sm.OLS(ximg,yimg_c)
    results = model.fit()
    coef_angular = results.params[1] # Pegamos o beta 1
    coef_linear =  results.params[0] # Pegamso o beta 0
    return coef_angular, coef_linear, pontos # Pontos foi adicionado para performance, como mencionado no notebook


def ajuste_linear_grafico_x_fy(mask_in, print_eq = False): 
    """Faz um ajuste linear e devolve uma imagem rgb com aquele ajuste desenhado sobre uma imagem
       Trabalhando com x em funcão de y
    """

    # vamos criar uma imagem com 50% do tamanho para acelerar a regressao 
    # isso nao afeta muito o angulo

    scale_percent = 50 # percent of original size
    width = int(mask_in.shape[1] * scale_percent / 100)
    height = int(mask_in.shape[0] * scale_percent / 100)
    dim = (width, height)

    mask = cv2.resize(mask_in, dim)

    coef_angular, coef_linear, pontos  = ajuste_linear_x_fy(mask)
    if print_eq: 
        print("x = {:3f}*y + {:3f}".format(coef_angular, coef_linear))
    ximg = pontos[1]
    yimg = pontos[0]
    y_bounds = np.array([min(yimg), max(yimg)])
    x_bounds = coef_angular*y_bounds + coef_linear
    # print("x bounds", x_bounds)
    # print("y bounds", y_bounds)
    x_int = x_bounds.astype(dtype=np.int64)
    y_int = y_bounds.astype(dtype=np.int64)
    mask_bgr =  cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)    
    cv2.line(mask_bgr, (x_int[0], y_int[0]), (x_int[1], y_int[1]), color=(0,0,255), thickness=11);    
    return mask_bgr, coef_angular, coef_linear


## Fim do código vindo do notebook


##  Código vindo da funcão cormodule.py da APS 4 

import rospy
import numpy as np
import tf
import math
import cv2
import time
from geometry_msgs.msg import Twist, Vector3, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import smach
import smach_ros


def identifica_cor(frame, cor):
    '''
    Segmenta o maior objeto cuja cor é parecida com cor_h (HUE da cor, no espaço HSV).
    '''

    # No OpenCV, o canal H vai de 0 até 179, logo cores similares ao 
    # vermelho puro (H=0) estão entre H=-8 e H=8. 
    # Precisamos dividir o inRange em duas partes para fazer a detecção 
    # do vermelho:
    # frame = cv2.flip(frame, -1) # flip 0: eixo x, 1: eixo y, -1: 2 eixos
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if cor == 'blue':
        cor_menor = np.array([78, 50, 50])
        cor_maior = np.array([120, 255, 255])

    elif cor == 'green':
        cor_menor = np.array([35, 50, 50])
        cor_maior = np.array([70, 255, 255])

    elif cor == 'orange':
        cor_menor = np.array([0, 200, 100])
        cor_maior = np.array([10, 255, 255])


    segmentado_cor = cv2.inRange(frame_hsv, cor_menor, cor_maior)

    # Note que a notacão do numpy encara as imagens como matriz, portanto o enderecamento é
    # linha, coluna ou (y,x)
    # Por isso na hora de montar a tupla com o centro precisamos inverter, porque 
    centro = (frame.shape[1]//2, frame.shape[0]//2)


    def cross(img_rgb, point, color, width,length):
        cv2.line(img_rgb, (int( point[0] - length/2 ), point[1] ),  (int( point[0] + length/2 ), point[1]), color ,width, length)
        cv2.line(img_rgb, (point[0], int(point[1] - length/2) ), (point[0], int( point[1] + length/2 ) ),color ,width, length) 



    # A operação MORPH_CLOSE fecha todos os buracos na máscara menores 
    # que um quadrado 7x7. É muito útil para juntar vários 
    # pequenos contornos muito próximos em um só.
    segmentado_cor = cv2.morphologyEx(segmentado_cor,cv2.MORPH_CLOSE,np.ones((7, 7)))

    # Encontramos os contornos na máscara e selecionamos o de maior área
    #contornos, arvore = cv2.findContours(segmentado_cor.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)	
    contornos, arvore = cv2.findContours(segmentado_cor.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    maior_contorno = None
    maior_contorno_area = 0

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area > maior_contorno_area:
            maior_contorno = cnt
            maior_contorno_area = area

    # Encontramos o centro do contorno fazendo a média de todos seus pontos.
    if not maior_contorno is None :
        cv2.drawContours(frame, [maior_contorno], -1, [0, 0, 255], 5)
        maior_contorno = np.reshape(maior_contorno, (maior_contorno.shape[0], 2))
        media = maior_contorno.mean(axis=0)
        media = media.astype(np.int32)
        cv2.circle(frame, (media[0], media[1]), 5, [0, 255, 0])
        cross(frame, centro, [255,0,0], 1, 17)
    else:
        media = (0, 0)

    # Representa a area e o centro do maior contorno no frame
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    cv2.putText(frame," CENTRO COR ({:d},{:d})".format(*media),(0,80), 1, 1, (255,255,255),1,cv2.LINE_AA)
    cv2.putText(frame," AREA COR {:0.1f}".format(maior_contorno_area),(0,100), 1, 1 , (255,255,255), 1 ,cv2.LINE_AA)
    
    # cv2.imshow('Filtra Creeper', frame)

    return media, centro, maior_contorno_area


def aruco_reader(cv_image,ids,corners,marker_size,camera_matrix,camera_distortion,font):

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
            # str_position = "Marker x=%4.0f  y=%4.0f  z=%4.0f"%(tvec[0], tvec[1], tvec[2])
            #print(str_position)
            # cv2.putText(cv_image, str_position, (0, 75), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

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
            # str_dist = "Dist aruco=%4.0f  dis.np=%4.0f"%(distance, distancenp)
            #print(str_dist)
            # cv2.putText(cv_image, str_dist, (0, 15), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

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
            # print(str_distfocal)
            # cv2.putText(cv_image, str_distfocal, (0, 30), font, 1, (255, 255, 255), 1, cv2.LINE_AA)	



