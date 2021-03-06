import cv2
import imutils
from imutils import contours
import numpy as np


# rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,4))
# sqr_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))

# for index in range(1,9):
# 	file_name = 'cheque' + str(index) + '.jpg'

# 	# Caarregar imagem
# 	im = cv2.imread(file_name)
# 	im = imutils.resize(im, width = 900)

# 	cv2.imshow(file_name, im)
# 	cv2.waitKey(0)

# 	# Converter pra grayscale
# 	im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# 	# Faz uma op. morfologica Black Hat => acha coisas pretas em fundo claro
# 	im_blackhat = cv2.morphologyEx(im_gray, cv2.MORPH_BLACKHAT, rect_kernel)

# 	# Fazer o gradiente em x e normaliza-lo entre [0,255]
# 	grad_x = cv2.Sobel(im_blackhat,
# 			 ddepth = cv2.CV_32F,
# 			 dx = 1, dy = 0,
# 			 ksize = -1)
# 	grad_x = np.absolute(grad_x)

# 	(min_grad, max_grad) = (np.min(grad_x), np.max(grad_x))
# 	grad_x = 255 * ((grad_x - min_grad) / (max_grad - min_grad))
# 	grad_x = grad_x.astype('uint8')

# 	# Fechamento - Binarização - Fechamento
# 	#   Fecha os buracos fazendo com que os caracteres formem um bloco
# 	im_cont = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)
# 	thresh = cv2.threshold(im_cont, 0, 255,
# 		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# 	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqr_kernel)

# 	cv2.imshow(file_name, thresh)
# 	cv2.waitKey(0)

# 	# Acha todos os contornos dos blocos da imagem e os ordena da esquerda pra direita
# 	conts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 	conts = conts[0] if imutils.is_cv2() else conts[1]
# 	conts = contours.sort_contours(conts, method="left-to-right")[0]
# 	locs = []

# 	# Limites para filtragem da ROI
# 	# 	Obtidos empiricamente
# 	V_limit = 300
# 	H_l_limit = 10
# 	skip_flag = False

# 	ar_l_limit = 8
# 	ar_h_limit = 25

# 	# Desenhar os limites para visualizacao
# 	cv2.line(im, (0, V_limit), (900, V_limit), (0,0,255), 3)
# 	cv2.rectangle(im, (50, V_limit - 50), (50 + (ar_l_limit * H_l_limit), V_limit - 50 + H_l_limit), (0,0,255), 2)
# 	cv2.rectangle(im, (50, V_limit - 50), (50 + (ar_h_limit * H_l_limit), V_limit - 50 + H_l_limit), (0,255,255), 2)

	
# 	# Loop passando por todos os contornos da imagem
# 	for (i, contorno) in enumerate(conts):
# 		if skip_flag:
# 			skip_flag = False
# 			continue

# 		# Feito o retangulo do contorno
# 		(x, y, w, h) = cv2.boundingRect(contorno)
# 		aspect_ratio = w/float(h)

# 		# Operacao booleana para verificar a validade da ROI
# 		# low_limit = y > V_limit
# 		# width_limit = w < W_h_limit and w > W_l_limit
# 		# height_limit = h > H_l_limit

# 		# in_limit = low_limit and width_limit and height_limit

# 		ar_limit = (aspect_ratio <= ar_h_limit) and (aspect_ratio >= ar_l_limit)
# 		in_limit = ar_limit and (y > V_limit)


# 		if(in_limit):
# 			#if(w < W_h_limit/2):
# 			#	w = W_h_limit
# 			#	skip_flag = True
# 			cv2.rectangle(im, (x,y), (x+w,y+h), (255,0,0), 2)
# 			locs.append((x,y,w,h))


# 	cv2.imshow(file_name, im)
# 	cv2.waitKey(0)


# Como Funcao
def getROI(imagem, ar_l_limit = 8, ar_h_limit = 25, V_limit = 300):

	rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,4))
	sqr_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))

	# Caarregar imagem
	im = imutils.resize(imagem, width = 900)

	# Converter pra grayscale
	im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	# Faz uma op. morfologica Black Hat => acha coisas pretas em fundo claro
	im_blackhat = cv2.morphologyEx(im_gray, cv2.MORPH_BLACKHAT, rect_kernel)

	# Fazer o gradiente em x e normaliza-lo entre [0,255]
	grad_x = cv2.Sobel(im_blackhat,
			 ddepth = cv2.CV_32F,
			 dx = 1, dy = 0,
			 ksize = -1)
	grad_x = np.absolute(grad_x)

	(min_grad, max_grad) = (np.min(grad_x), np.max(grad_x))
	grad_x = 255 * ((grad_x - min_grad) / (max_grad - min_grad))
	grad_x = grad_x.astype('uint8')

	# Fechamento - Binarização - Fechamento
	#   Fecha os buracos fazendo com que os caracteres formem um bloco
	im_cont = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)
	thresh = cv2.threshold(im_cont, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqr_kernel)


	# Acha todos os contornos dos blocos da imagem e os ordena da esquerda pra direita
	conts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	conts = conts[0] if imutils.is_cv2() else conts[1]
	conts = contours.sort_contours(conts, method="left-to-right")[0]
	locs = []

		# Loop passando por todos os contornos da imagem
	for (i, contorno) in enumerate(conts):

		# Feito o retangulo do contorno
		(x, y, w, h) = cv2.boundingRect(contorno)
		aspect_ratio = w/float(h)

		ar_limit = (aspect_ratio <= ar_h_limit) and (aspect_ratio >= ar_l_limit)
		in_limit = ar_limit and (y > V_limit)

		if(in_limit):
			cv2.rectangle(im, (x,y), (x+w,y+h), (255,0,0), 2)
			locs.append((x,y,w,h))

	return locs

def assign_to_nearest(img, centers):
	centers_dict = {i : [] for i, _ in enumerate(centers)}
	height, width = img.shape

	for j in range(0,height):
		for i in range(0, width):
			if(img[j][i] != 0):

				dist = 1000
				index = 0
				for k, center in enumerate(centers):
					new_dist = np.power(i - center[0], 2) + np.power(j - center[1], 2)
					if new_dist < dist:
						dist = new_dist
						index = k
				centers_dict[index].append([i,j])

	return centers_dict

def update_centers(centers_dict):
	indexes = centers_dict.keys()
	new_centers = []

	for index in indexes:
		new_center_x = 0
		new_center_y = 0

		for pixel in centers_dict[index]:
			new_center_x = new_center_x + pixel[0]
			new_center_y = new_center_y + pixel[1]

		new_center_x = int(new_center_x / len(centers_dict[index]))
		new_center_y = int(new_center_y / len(centers_dict[index]))

		new_centers.append([new_center_x, new_center_y])

	return new_centers

def kmeans(img, n_classes ,iterations= 3):

	rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,4))
	sqr_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))

	# Caarregar imagem
	im = imutils.resize(img, width = 300)

	# Converter pra grayscale
	im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	# Faz uma op. morfologica Black Hat => acha coisas pretas em fundo claro
	im_blackhat = cv2.morphologyEx(im_gray, cv2.MORPH_BLACKHAT, sqr_kernel)

	# Fechamento - Binarização - Fechamento
	#   Fecha os buracos fazendo com que os caracteres formem um bloco
	thresh = cv2.threshold(im_blackhat, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kernel)

	height, width = thresh.shape
	digit_size = width/n_classes
	center_y = height/2
	center_xs = [(i + 0.5) * digit_size for i in range(0, n_classes)]

	centers = [[x, center_y] for x in center_xs]

	for _ in range(0, iterations):
		centers_dict = assign_to_nearest(thresh, centers)

		centers = update_centers(centers_dict)

	return centers_dict


for index in range(3,6):
	file_name = '/Users/nicolautahan/Documents/GitHub/BinarizarCheque/cheque' + str(index) + '.jpg'

	img = cv2.imread(file_name)
	img = imutils.resize(img, width = 900)

	c = getROI(img)
	num_digits = [10, 11, 13]

	for (x,y,w,h), classes in zip(c, num_digits):
		#cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
		cropped_img = img[y-5:y+h+5, x-5:x+w+5]
		cropped_img = imutils.resize(cropped_img, width = 300)

		a = kmeans(cropped_img, classes, 5)
		for index in a.keys():
			vec = np.array(a[index])
			x = min(vec[:,0])
			cv2.rectangle(cropped_img, (x, 0), (x, 200), (255,0,0), 4) 
		cv2.imshow('a', cropped_img)
		cv2.waitKey(0)

		


