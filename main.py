import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import math
import sys
from fen import board2fen,compareFen,PIECE_TYPES




def clasify(img,piece):
    def getSquare(i,img):
        res = img.copy()
        SQUARE_SIZE = 60
        y = (i // 8) * SQUARE_SIZE 
        x = (i % 8) * SQUARE_SIZE
        return res[y:y+SQUARE_SIZE, x:x+SQUARE_SIZE]
    
    templates = 'templates'

    img = cv.Canny(img, 60, 80)
    img = cv.distanceTransform(img, cv.DIST_L2, 5)
    
    img = getSquare(piece,img)
    img2 = img.copy()

    
    methods = ['cv.TM_CCORR_NORMED']
    maximos = {'cv.TM_CCORR_NORMED':[-math.inf,"None"]}

    for meth in methods:
        for clase in os.listdir(templates):
            path = os.path.join(templates, clase)
            if os.path.isdir(path):
                for templatePath in os.listdir(path):
                    template = cv.imread(os.path.join(templates, clase,templatePath),0)   
                    template = cv.Canny(template, 60, 80)
                    template = cv.distanceTransform(template, cv.DIST_L2, 5)
                    img = img2.copy()
                    method = eval(meth)
                    # Apply template Matching with diferrent rotations of tempalte
                    for i in [-2,-1,0,1]:
                        if i != -2:
                            template = cv.flip(template, i)
                        res = cv.matchTemplate(img,template,method)
                        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                        if max_val > maximos[meth][0]:
                            maximos[meth] = [max_val,clase]
    return maximos['cv.TM_CCORR_NORMED'][1].split(' ')[0]




def detectPieces(imgPath,expectedFen):
    from detectPieceColor import get_board_matrix
 
    img = cv.imread(imgPath,0)
    img = cv.resize(img, (480,480), interpolation = cv.INTER_AREA)

    board = get_board_matrix(imgPath)

    i = 0
    boardFen = []
    for r in range(8):
        row = []
        for c in range(8):
            if board[r][c] != -1: # There is a piece in this square
                piece = clasify(img,i)
                if board[r][c] == 0: # The piece is white
                    piece = piece.upper()
                row.append(piece)
            else:
                row.append('_')
            i +=1
        boardFen.append(row)
    
    fen = board2fen(boardFen)

    acc,confusionMatrix = compareFen(fen,expectedFen)

    return acc,confusionMatrix,fen

def plotConfussionMatrix(cm):
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt     
    df_cm = pd.DataFrame(cm, index = [i for i in PIECE_TYPES],
                  columns = [i for i in PIECE_TYPES])
    plt.figure(figsize = (15,10))
    sn.heatmap(df_cm, annot=True,fmt='d')   
    plt.show()

def openGameInBrowser(fen):
    import webbrowser

    url = 'https://lichess.org/editor/' + fen
    # Path to Chrome
    chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'

    webbrowser.get(chrome_path).open(url)

def detect():
    imgPath = sys.argv[2]
    fenPath = os.path.join(imgPath.replace(imgPath.split('\\')[-1],""), 'board.fen')
    if os.path.isfile(fenPath):
        f = open(fenPath, "r")
        fen = f.readline()
        f.close() 
        acc,cm,predictedfen = detectPieces(imgPath,fen)
        print("Accuracity:", acc)
        plotConfussionMatrix(cm)
        openGameInBrowser(fen)
        print(fen)
    else:
        _,_,predictedfen = detectPieces(imgPath,'8/8/8/8/8/8/8/8') 

    openGameInBrowser(predictedfen)

def test():
    acc = []
    confussionMatrix = np.zeros((len(PIECE_TYPES),len(PIECE_TYPES)),np.uint8)
    if len(sys.argv) > 2:
        maxImages = int(sys.argv[2])
    else:
        maxImages = 30
    count = 0
    stop = False
    root = r'raw\1'
    for folder in os.listdir(root):
        imagePath = os.listdir(os.path.join(root,folder))[0]
        if imagePath.endswith('.jpg'):
            count +=1
            f = open(os.path.join(root,folder, 'board.fen'), "r")
            fen = f.readline()
            f.close() 
            accAux,confussionMatrixAux,_ = detectPieces(os.path.join(root,folder,imagePath),fen)
            print("Image:",count)
            print('Acc:', accAux)
            acc.append(accAux)
            confussionMatrix = confussionMatrix + confussionMatrixAux
            
            if count>maxImages-1:
                stop = True
                break
        
    acc = np.mean(acc)

    print("Mean Accuracity:",acc)
    plotConfussionMatrix(confussionMatrix)



if __name__ == '__main__':

    mode = sys.argv[1]

    if mode == 'detect':
        detect()
    elif mode == 'test':
        test()
    else:
        print("Ningún modo especificado")
    