import cv2
import matplotlib.pyplot as plt 
import numpy as np

img = cv2.imread('circulos\circulos_1.png')
copia = img.copy()
img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

circulos = cv2.HoughCircles(
    img_cinza,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=80,       
    param1=100,       
    param2=50,       
    minRadius=40,    
    maxRadius=60
)

if circulos is not None:
    circulos = np.round(circulos[0, :]).astype("int")

    for (x, y, r) in circulos:
        cv2.circle(copia, (x, y), r, (0, 255, 0), 4)  # 
        cv2.rectangle(copia, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)  

plt.figure(figsize=(8, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Imagem Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(copia, cv2.COLOR_BGR2RGB))
qtd = len(circulos) if circulos is not None else 0
plt.title(f"Círculos Detectados: {qtd}")
plt.axis('off')


plt.tight_layout()
plt.show()
