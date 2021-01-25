import cv2
import numpy as np
#import matplotlib.pyplot as plt
#from pprint import pprint

tam_casilla = 60
porcentaje_mismo_color = 0.95
    
TIPO0 = [['B','N'],
         ['N','B']]

TIPO1 = [['N','B'],
         ['B','N']]

RESIZE = 10
THRESHOLD = 5000

# COLOR #
KERNEL = np.ones((3, 3), np.float32)
# HSV Color Ranges
GREEN_LOWER = (40,100,45)
GREEN_UPPER = (120,255,255)
ORANGE_LOWER = (0,100,64)
ORANGE_UPPER = (20,255,255)

def get_colors(c):
    n_255, n_0 = np.count_nonzero(c == 255),np.count_nonzero(c == 0)
    return (n_255,n_0)
 
def get_board_type(img_binary):
    color = 'X'
    fin = False
    x = 0
    y = 0
    
    for row in range(8):
        if fin:
            break
        
        y = tam_casilla*row
        x = 0
        for col in range(8):
            c = img_binary[y:y+tam_casilla, x:x+tam_casilla]
            x += tam_casilla
            n_255, n_0 = get_colors(c)
            casilla = [row, col]
            
            if n_255 > (tam_casilla**2 * porcentaje_mismo_color):
                # print("casilla blanca vacia: ", casilla)
                color = 'B'
                fin = True
            elif n_0 > (tam_casilla**2 * porcentaje_mismo_color):
                # print("casilla negra vacia: ", casilla)
                color = 'N'
                fin = True
            
            if fin:
                break
        
    if color == 'X':
        raise NameError("get_board_type: No se ha podido identificar el color de ninguna casilla")

    if TIPO0[casilla[0]%2][casilla[1]%2] == color:
        tipo = 0
    else:
        tipo = 1
        
    return tipo

def mask(img, lower, upper):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, KERNEL)
        return closing
    
def get_square_type(casilla):
    # Devuelve el tipo de una casilla:
    # 0: pieza blanca
    # 1: pieza negra
    # -1: sin pieza
    
    greenMask = mask(casilla, GREEN_LOWER, GREEN_UPPER)
    orangeMask = mask(casilla, ORANGE_LOWER, ORANGE_UPPER)

    if np.sum(greenMask) < THRESHOLD and np.sum(orangeMask) < THRESHOLD:
        color = -1
    else:
        color = 1 if (np.sum(greenMask) > np.sum(orangeMask)) else 0
    
    return color

def get_squares(img):
    tablero = []
    x = 0
    y = 0
    
    for row in range(8):       
        y = tam_casilla*row
        x = 0
        tipos = []
        for col in range(8):
            casilla = img[y+RESIZE:y+tam_casilla-RESIZE, x+RESIZE:x+tam_casilla-RESIZE]
            casilla = cv2.cvtColor(casilla, cv2.COLOR_BGR2RGB)
            tipo = get_square_type(casilla)
            x += tam_casilla
            tipos.append(tipo)
        tablero.append(tipos)
            
    return tablero


def get_board_matrix(fichero):
    
    img_original = cv2.imread(fichero)
    img_original = cv2.resize(img_original, (480,480), interpolation = cv2.INTER_AREA)
    img = cv2.imread(fichero,0)
    img = cv2.resize(img, (480,480), interpolation = cv2.INTER_AREA)

    blur = cv2.GaussianBlur(img,(5,5),0)
    _, img_binary = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    tablero = get_squares(img_original)
    
    tipo = get_board_type(img_binary)
    if tipo == 1:
        l = tablero.copy()
        #tablero =[list(i) for i in zip(*l)]
        
        
    return tablero
