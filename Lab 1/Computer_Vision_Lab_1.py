#!/usr/bin/env python
# coding: utf-8

# # ΟΡΑΣΗ ΥΠΟΛΟΓΙΣΤΩΝ - ΕΡΓΑΣΤΗΡΙΑΚΗ ΑΣΚΗΣΗ 1
# 
# Δωροθέα Κουμίδου 03119712
# 
# Γιώργος Χαραλάμπους 03119706
# 
# 
# Η εργαστηριακή άσκηση part1&2 πραγματοποιήθηκε με τη βοήθεια του google colab.
# 

# ### Μέρος 1. Ανίχνευση Ακμών σε Γκρίζες Εικόνες

# 1.1 Δημιουργία εικόνων εισόδου

# In[2]:


get_ipython().system('pip install matplotlib')


# In[3]:


#1.1.1
import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("edgetest_24.png",cv2.IMREAD_GRAYSCALE)

#normalize image to [0,1]
image = image.astype(np.double)/255

img_height, img_width = image.shape
plt.imshow(image,cmap='gray')
print(image.shape)


# In[4]:


#1.1.2
#υπολογισμός τυπικής απόκλισης μέσω PSNR
import numpy as np

def sigma_n(PSNR: int,
          img_max:int,
          img_min:int):
  return ((img_max - img_min)/(10**(PSNR/20)))

img_max = image.max()
img_min = image.min()

#image 1: PSNR = 20 dB
PSNR1 = 20
image1 = image + np.random.normal(0,sigma_n(PSNR1,img_max,img_min),size=(img_height,img_width))
plt.title("PSNR = 20 dB")
plt.imshow(image1,cmap='gray')
plt.show()

#image 2: PSNR = 10 dB
PSNR2 = 10
image2 = image + np.random.normal(0,sigma_n(PSNR2,img_max,img_min),size=(img_height,img_width))
plt.title("PSNR = 10 dB")
plt.imshow(image2,cmap='gray')
plt.show()


# 1.2 ΥΛΟΠΟΙΗΣΗ ΑΛΓΟΡΙΘΜΩΝ ΑΝΙΧΝΕΥΣΗΣ ΑΚΜΩΝ

# In[5]:


#1.2.1

#Smoothing
sigma = 1.5
n = int(2*np.ceil(3*sigma)+1)

G1D = cv2.getGaussianKernel(n,sigma) #1D gaussian filter
G2D = G1D @ G1D.T #symmetric 2D gaussian filter


def LoG(n,sigma):
  x = np.linspace(-n/2,n/2,n)
  y = np.linspace(-n/2,n/2,n)
  X,Y = np.meshgrid(x,y)

  Z = (X**2 + Y**2-2*sigma**2)/(2*np.pi*sigma**6)
  Z *= np.exp(-(X**2+Y**2)/(2*sigma**2))
  return Z


#1.2.2
#The 2 types od Laplacian
L1 = cv2.filter2D(image1,-1,LoG(n,sigma))
plt.imshow(L1,cmap='gray')
plt.title("Linear Laplacian of image (PSNR=20dB)")
plt.show()

#Non-Linear (L2)
kernel = np.array([[0,1,0],
                  [1,1,1],
                  [0,1,0]], dtype=np.uint8)

smoothed_img = cv2.filter2D(image1,-1,G2D)

L2 = cv2.dilate(smoothed_img,kernel)+cv2.erode(smoothed_img,kernel)-2*smoothed_img

plt.imshow(L2,cmap='gray')
plt.title("Non-linear Laplacian of image (PSNR = 20dB)")
plt.show()

plt.imshow(smoothed_img,cmap='gray')
plt.title("Smoothed image (PSNR = 20dB)")
plt.show()


# In[6]:


#1.2.3
X = (L2 >= 0) #pick only those pixels with non-zero values
X = X.astype(np.uint8)
plt.imshow(X,cmap="gray")
plt.title("X = (L2>=0)")
plt.show()


X = (L1 >= 0) #pick only those pixels with non-zero values
X = X.astype(np.uint8)
plt.imshow(X,cmap="gray")
plt.title("X = (L1>=0)")
plt.show()

#find image's zerocrossings
Y = cv2.dilate(X,kernel) - cv2.erode(X,kernel)
plt.imshow(Y,cmap="gray")
plt.title("Zerocrossings")
plt.show()


# In[7]:


#1.2.4
#Threshold on Zerocrossing
height, width = Y.shape

img_gradient = np.array(np.gradient(smoothed_img))
theta = 0.2

#find the max values of both axis gradient
max1 = np.absolute(img_gradient[0]).max()
max2 = np.absolute(img_gradient[1]).max()

#find edges of both axis gradients
edges1 = (np.logical_and(Y==1,np.absolute(img_gradient[0])>theta*max1)).astype(int)
edges2 = (np.logical_and(Y==1,np.absolute(img_gradient[1])>theta*max2)).astype(int)

result = np.logical_or(edges1,edges2)

plt.imshow(edges1,cmap='gray')
plt.show()

plt.imshow(edges2,cmap='gray')
plt.show()

plt.imshow(result,cmap='gray')
plt.show()


# In[8]:


def EdgeDetect(image:np.array,
               sigma:int,
               theta:int,
               type_of_filter:str):
  assert type_of_filter == 'linear' or type_of_filter == 'non-linear','Type should be "linear" or "non-linear"'

  #1.2.1
  #Filter Creation
  n = int(2*np.ceil(3*sigma)+1) #n: kernel size (n x n)

  gauccian_1D = cv2.getGaussianKernel(n,sigma)
  gauccian_2D = gauccian_1D @ gauccian_1D.T   #2D and symmetric gauccian filter

  def LoG(n,sigma): #return Laplacian-of-Gauccian filter
    x = np.linspace(-n/2,n/2,n)
    y = np.linspace(-n/2,n/2,n)
    X,Y = np.meshgrid(x,y)

    Z = (X**2 + Y**2-2*sigma**2)/(2*np.pi*sigma**6)
    Z *= np.exp(-(X**2+Y**2)/(2*sigma**2))
    return Z

  log = LoG(n,sigma)

  #1.2.2
  #Laplacian L (smoothing the image)
  kernel = np.array([[0,1,0],
                  [1,1,1],
                  [0,1,0]], dtype=np.uint8)
  blur_img = cv2.filter2D(image,-1,gauccian_2D)

  #choose between L1 or L2 smoothing methods
  if type_of_filter == 'linear':
    L = cv2.filter2D(image,-1,log) #convolution of log and image
  elif type_of_filter == 'non-linear':
    L = cv2.dilate(blur_img,kernel)+cv2.erode(blur_img,kernel)-2*blur_img

  #1.2.3
  #Locating zerocrossings
  X = (L>=0)  #binary image
  X = X.astype(np.uint8)

  Y = cv2.dilate(X,kernel) - cv2.erode(X,kernel)

  #1.2.4
  #Finding the edges (threshold on zerocrossings)
  img_gradient = np.array(np.gradient(blur_img))

  #find the max values of both axis gradient
  max1 = np.absolute(img_gradient[0]).max()
  max2 = np.absolute(img_gradient[1]).max()

  #find edges of both axis gradients
  edges1 = (np.logical_and(Y==1,np.absolute(img_gradient[0])>theta*max1)).astype(int)
  edges2 = (np.logical_and(Y==1,np.absolute(img_gradient[1])>theta*max2)).astype(int)

  result = np.logical_or(edges1,edges2).astype(int)

  return result


# In[9]:


#Testing EdgeDetect on 2 different images
edges1 = EdgeDetect(image1,1.5,0.2,'linear') #Edges using PSNR=20dB
edges2 = EdgeDetect(image2,3,0.2,'linear')  #Edges using PSNR=10dB

fig,axis = plt.subplots(2,2,figsize=(10,10))
axis[0,0].imshow(edges1,cmap = "gray")
axis[0,0].set_title("linear with PSNR=20dB")
axis[0,1].imshow(edges2,cmap = "gray")
axis[0,1].set_title("linear with PSNR=10dB")

edges1 = EdgeDetect(image1,1.5,0.2,'non-linear') #Edges using PSNR=20dB
edges2 = EdgeDetect(image2,3,0.2,'non-linear')  #Edges using PSNR=10dB

axis[1,0].imshow(edges1,cmap = "gray")
axis[1,0].set_title("non-linear with PSNR=20dB")

axis[1,1].imshow(edges2,cmap = "gray")
axis[1,1].set_title("non-linear with PSNR=10dB")



# 1.3 ΑΞΙΟΛΟΓΗΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ΑΝΙΧΝΕΥΣΗΣ ΑΚΜΩΝ
# 

# In[10]:


#1.3.1
#Finding "real" edges
def real_edges_detect(image,theta_real):
  kernel = np.array([[0,1,0],
                  [1,1,1],
                  [0,1,0]], dtype=np.uint8)

  M = cv2.dilate(image,kernel) - cv2.erode(image,kernel)
  T = M>theta_real
  T = T.astype(np.uint8)
  return T

real_edges = real_edges_detect(image,0.1)

plt.imshow(real_edges,cmap = "gray")
plt.title('"Real" edges of original image')


# In[11]:


#1.3.2
def detection_quality(real_edges,
                      my_edges):
  precision = (real_edges & my_edges).sum()/real_edges.sum()
  recall = (real_edges & my_edges).sum()/my_edges.sum()

  return (precision+recall)/2

print("Η ποσοτική αξιολόγηση για PSNR=10dB είναι:", detection_quality(real_edges,edges2))
print("Η ποσοτική αξιολόγηση για PSNR=20dB είναι:", detection_quality(real_edges,edges1))


# In[12]:


#1.3.3
def testing(image, PSNR, theta_edge, theta_real,sigma,filter_type):
  img_height, img_width = image.shape
  img_max = image.max()
  img_min = image.min()
  noise_sigma = sigma_n(PSNR,img_max,img_min)

  #apply noise to the image
  noised_img = image + np.random.normal(0,noise_sigma,size=(img_height,img_width))

  #find edges of noised image
  noised_edges = EdgeDetect(noised_img,sigma,theta_edge,filter_type)

  #find real edges
  real_edges = real_edges_detect(image,theta_real)

  #calculate quality
  quality = detection_quality(real_edges,noised_edges)
  q = round(quality,3)
  fig ,axis = plt.subplots(1,2,figsize = (10,5))
  axis[0].imshow(real_edges,cmap="gray_r")
  axis[0].set_title("Real edges")
  axis[1].imshow(noised_edges,cmap = "gray_r")
  axis[1].set_title(f"Edges of PSNR={PSNR}dB using {filter_type} filter")
  fig.suptitle(f"Edge detection of quality={q}, sigma={sigma}, theta edge={theta_edge}")
  plt.show()

  return quality

#testing(image,PSNR=10,theta_edge=0.2,theta_real=0.1,sigma=1.5,filter_type='non-linear')


# In[13]:


#1.3.3
#Testing session

#Test several sigma,theta,PSNR on L1 filtering
theta_real = 0.2

filters = ['linear','non-linear']
sigmas = [1.5,2,2.5,3]
PSNR = [10,20]
thetas = [ 0.15,0.2, 0.25]

for f in filters:
  for s in sigmas:
    for p in PSNR:
      for th in thetas:
        q = testing(image=image,PSNR=p,theta_edge=th,theta_real=theta_real,sigma=s,filter_type=f)


# 1.4 ΕΦΑΡΜΟΓΗ ΣΕ ΠΡΑΓΜΑΤΙΚΕΣ ΕΙΚΟΝΕΣ

# In[14]:


real_image = cv2.imread("butterfly.jpg")
gray_image = cv2.imread("butterfly.jpg",cv2.IMREAD_GRAYSCALE)

gray_image = gray_image.astype(np.uint8)/255

sigma = 3
theta = 0.2
filter = 'non-linear'

detected_edges = EdgeDetect(gray_image,sigma=sigma,theta=theta,type_of_filter=filter)

fig,axis = plt.subplots(1,3,figsize=(15,4))
axis[0].imshow(real_image)
axis[0].set_title("Image")
axis[1].imshow(gray_image,cmap="gray")
axis[1].set_title("Image on grayscale")
axis[2].imshow(detected_edges,cmap='gray')
axis[2].set_title("Detected edges")
fig.suptitle(f"Using filter {filter},sigma={sigma},theta={theta}")
plt.show()


# In[15]:


#Testing for several sigmas and thetas
thetas = [0, 0.1, 0.2 ,0.5, 0.8, 1 ]
sigmas = [ 1 ,1.5, 3, 5 ]
filter = ['linear','non-linear']

real_edges = real_edges_detect(gray_image, 0.1)
best_quality = 0

for f in filters:
  fig, axis = plt.subplots(len(sigmas),len(thetas),figsize=(30,20))
  for i,s in enumerate(sigmas):
    for j,th in enumerate(thetas):
      detected_edges = EdgeDetect(gray_image,sigma=s,theta=th,type_of_filter=f)
      quality = detection_quality(real_edges,detected_edges)
      if quality >= best_quality:
          best_quality = quality
          best_theta=th
          best_sigma=s
          best_filter=f
      axis[i][j].imshow(detected_edges,cmap='gray')
      axis[i][j].set_title(f"Quality={round(quality,2)}, Theta={th}, Sigma={s}")
  fig.suptitle(f"Using {f} filter type")
  plt.show()
  print('\n')


# In[17]:


#our best result
best_edges = EdgeDetect(gray_image,sigma=best_sigma,theta=best_theta,type_of_filter=best_filter)
plt.imshow(best_edges,cmap='gray')
plt.title(f"{best_filter} filter, sigma={best_sigma}, theta={best_theta} with quality {np.round(best_quality,3)}")


# ### Μέρος 2. Ανίχνευση σημείων ενδιαφέροντος

# 2.1 Ανίχνευση Γωνιών (χρήση Harris Detector)

# In[16]:


#2.1.1
sigma = 2 #differention factor
ns = int(2*np.ceil(3*sigma)+1)
gauccian_1D = cv2.getGaussianKernel(ns,sigma)
gauccian_2D = gauccian_1D @ gauccian_1D.T

butterfly = cv2.filter2D(gray_image,-1,gauccian_2D)
gradient_butterfly = np.gradient(butterfly) #[0]: x axis deriv, [1]: y axis deriv

p = 2.5 #integration factor
n_p=int(2*np.ceil(3*p)+1)
gp_1D = cv2.getGaussianKernel(n_p,p)
gp_2D = gp_1D @ gp_1D.T

#calculating J1,J2,J3 of J tensor
J1 = cv2.filter2D((gradient_butterfly[0]*gradient_butterfly[0]),-1,gp_2D)
J2 = cv2.filter2D((gradient_butterfly[0]*gradient_butterfly[1]),-1,gp_2D)
J3 = cv2.filter2D((gradient_butterfly[1]*gradient_butterfly[1]),-1,gp_2D)


# In[17]:


#2.1.2
#calculating the eigenvalues of J tensor
lambda_plus = (1/2)*(J1+J3+np.sqrt( (J1-J3)**2 + 4*J2**2 ))
lambda_minus = (1/2)*(J1+J3-np.sqrt( (J1-J3)**2 + 4*J2**2 ))

fig, axis = plt.subplots(1,2,figsize=(15,6))
axis[0].imshow(lambda_plus,cmap="gray")
axis[0].set_title("Lambda +")
axis[1].imshow(lambda_minus,cmap="gray")
axis[1].set_title("Lambda -")


# In[18]:


#Εισαγωγη βοηθητικού κωδικα
import sys
sys.path.append('/content/sample_data')
import cv24_lab1_part2_utils as utils
from cv24_lab1_part2_utils import disk_strel


# In[19]:


#2.1.3
#calculating the cornerness criterion
k = 0.05
R = lambda_minus*lambda_plus - k*(lambda_minus + lambda_plus)**2

#condition 1: pixels of R that are the max in a window determned by sigma

ns = int(np.ceil(3*sigma)*2+1)
B_sq = disk_strel(ns)
Cond1 = ( R == cv2.dilate(R,B_sq))

#condition 2: pixels (x,y) where R(x,y)>theta_corn*Rmax
theta_corn = 0.005
Rmax = R.max()
Cond2 = ( R >theta_corn*Rmax )


# In[20]:


def harris_stephens_detector(s,p,image,k):
  #calculate sigma gauccians
  n_s = int(2*np.ceil(3*s)+1)
  gs_1D = cv2.getGaussianKernel(n_s,s)
  gs_2D = gs_1D @ gs_1D.T

  #calculate p gauccians
  n_p = int(2*np.ceil(3*p)+1)
  gp_1D = cv2.getGaussianKernel(n_p,p)
  gp_2D = gp_1D @ gp_1D.T

  Is= cv2.filter2D(image,-1,gs_2D)
  gradient = np.gradient(Is)

  #calculating J1,J2,J3 of J tensor
  J1 = cv2.filter2D((gradient[0]*gradient[0]),-1,gp_2D)
  J2 = cv2.filter2D((gradient[0]*gradient[1]),-1,gp_2D)
  J3 = cv2.filter2D((gradient[1]*gradient[1]),-1,gp_2D)

  #calculating the eigenvalues of J tensor
  lambda_plus = (1/2)*(J1+J3+np.sqrt( (J1-J3)**2 + 4*J2**2 ))
  lambda_minus = (1/2)*(J1+J3-np.sqrt( (J1-J3)**2 + 4*J2**2 ))

  #calculating the cornerness criterion
  R = lambda_minus*lambda_plus - k*(lambda_minus + lambda_plus)**2

  #condition 1: pixels of R that are the max in a window determned by sigma
  ns = int(np.ceil(3*s)*2+1)
  B_sq = disk_strel(ns)
  Cond1 = ( R == cv2.dilate(R,B_sq))

  #condition 2: pixels (x,y) where R(x,y)>theta_corn*Rmax
  theta_corn = 0.05
  Rmax = R.max()
  Cond2 = ( R >theta_corn*Rmax )

  corners = np.logical_and(Cond1,Cond2)
  corners = corners.astype(int)
  rows,cols = corners.shape
  points = []
  for y in range(rows):
    for x in range(cols):
      if corners[y][x]>0:
          points.append([x,y,s])

  return np.array(points)


# Πιο πάνω φαίνεται ολόκληρη η διαδικασ'ια για την μια εικόνα μόνο. Παρακάτω γίνεται η εφαρμογή της ανίχνευσης πάνω σε όλες τις εικόνες του ερωτήματος

# In[23]:


from cv24_lab1_part2_utils import interest_points_visualization as visualise

butt = cv2.imread("butterfly.jpg")
butt_gray = cv2.imread("butterfly.jpg",cv2.IMREAD_GRAYSCALE)
points= harris_stephens_detector(s=2,p=2.5,k=0.05,image=butt_gray)

visualise(butt,points)

caravaggio= cv2.imread("Caravaggio.jpg")
caravaggio_gray = cv2.imread("Caravaggio.jpg",cv2.IMREAD_GRAYSCALE)
points= harris_stephens_detector(s=2,p=2.5,k=0.05,image=caravaggio_gray)

visualise(caravaggio,points)

urban= cv2.imread("urban_edges.jpg")
urban_gray = cv2.imread("urban_edges.jpg",cv2.IMREAD_GRAYSCALE)
points= harris_stephens_detector(s=2,p=2.5,k=0.05,image=urban_gray)

visualise(urban,points)


# 2.2 ΠΟΛΥΚΛΙΜΚΩΤΗ ΑΝΙΧΝΕΥΣΗ ΓΩΝΙΩΝ

# In[21]:


#2.2.1
#Finding the corners for multiple scaling factors
def find_all_corners(s0,p0,scale,N,image,k):
  all_scales = [(s0*scale**i,p0*scale**i) for i in range(N)] #create all scales
  corners_coord = []
  for s_i,p_i in all_scales:
    points = harris_stephens_detector(s=s_i,p=p_i,image=image,k=k)
    corners_coord.append(points)
  return corners_coord


# In[22]:


# 2.2.2
# Calculating normalised LoG for each image and sigma
def normalised_Log(image,sigma):
  n = int(2*np.ceil(3*sigma)+1)
  g1D = cv2.getGaussianKernel(n,sigma)
  g2D = g1D @ g1D.T
  L = cv2.filter2D(image,-1,g2D) #Laplaccian L of image
  Lx = np.gradient(L)[0]
  Ly = np.gradient(L)[1]
  Lxx = np.gradient(Lx)[0]
  Lyy = np.gradient(Ly)[1]
  return (sigma**2)*np.abs(Lxx+Lyy)

def find_all_LoG(image,s0,scale,N):
  all_sigmas = [s0*scale**i for i in range(N)]
  all_LoG = []
  for s in all_sigmas:
    img = normalised_Log(image,s)
    all_LoG.append(img)
  return all_LoG,all_sigmas


# In[23]:


def harris_laplace_detector(image,s0,p0,scale,N):
  print("Starting harris laplace...")
  all_logs,all_sigmas = find_all_LoG(image=image,s0=s0,scale=scale,N=N)
  print("LoGs and sigmas calculated")
  all_corners = find_all_corners(image=image,s0=s0,p0=p0,scale=s,N=N,k=0.05)
  print("Corners found")
  print("Starting calculating multiscalar corners...")
  result = [] #contains the coordinates of final edges
  for index in range(0,N):
    print(f"iteration number: {index+1} of {N}")
    corner_coord = all_corners[index]
    cur_log = all_logs[index]
    if(index>0 and index<N-1):
        prev_log = all_logs[index-1]
        next_log = all_logs[index+1]
    elif(index==0):
        prev_log = all_logs[index+1]
        next_log = all_logs[index+1]
    elif(index==N-1):
        prev_log = all_logs[index-1]
        next_log = all_logs[index-1]

    for p in corner_coord:
      x = int(p[0])
      y = int(p[1])
      if (cur_log[y][x]>prev_log[y][x]) and (cur_log[y][x]>next_log[y][x]):
        result.append([x,y,all_sigmas[index]])

  return np.array(result)


# In[45]:


s0=2
p0=1.5
N=4
s=1.5
res1 = harris_laplace_detector(image=caravaggio_gray,s0=s0,p0=p0,scale=s,N=N)
visualise(caravaggio,res1)
res2 = harris_laplace_detector(image=urban_gray,s0=s0,p0=p0,scale=s,N=N)
visualise(urban, res2)


# 
# 2.3 Ανίχνευση Blobs

# In[24]:


#2.3.1
def hessian_detector(image,sigma,theta_corn):
  n = int(2*np.ceil(3*sigma)+1)
  gauccian_1D = cv2.getGaussianKernel(n,s)
  gauccian_2D = gauccian_1D @ gauccian_1D.T
  Is = cv2.filter2D(image,-1,gauccian_2D) #blurred image

  #Calculate gradients
  Lx = np.gradient(Is)[0]
  Ly = np.gradient(Is)[1]
  Lxx = np.gradient(Lx)[0]
  Lyy = np.gradient(Ly)[1]
  Lxy = np.gradient(Lx)[1]

  #H = [[Lxx,Lxy],[Lxy,Lyy]] Hessian Matrix
  H = np.concatenate([np.concatenate([Lxx,Lyy],axis=1),np.concatenate([Lxy,Lyy],axis=1)],axis=0)
  R = Lxx*Lyy - (Lxy)**2
  Rmax = R.max()

  #condition 1: pixels of R that are the max in a window determned by sigma

  ns = int(np.ceil(3*sigma)*2+1)
  B_sq = disk_strel(ns)
  Cond1 = ( R == cv2.dilate(R,B_sq))

  #condition 2: pixels (x,y) where R(x,y)>theta_corn*Rmax
  Rmax = R.max()
  Cond2 = ( R >theta_corn*Rmax )

  corners = np.logical_and(Cond1,Cond2)
  h,w = corners.shape
  points = []
  for y in range(h):      #pick row
    for x in range(w):    #pick column
      if corners[y][x]>0:
        points.append([x,y,sigma])
  return np.array(points)


# In[35]:


caravaggio = cv2.imread("Caravaggio.jpg")
caravaggio_gray = cv2.imread("Caravaggio.jpg",cv2.IMREAD_GRAYSCALE)
points = hessian_detector(image=caravaggio_gray,sigma=1.5,theta_corn=0.01)
visualise(caravaggio,points)


# In[36]:


urban = cv2.imread("urban_edges.jpg")
urban_gray = cv2.imread("urban_edges.jpg",cv2.IMREAD_GRAYSCALE)
points = hessian_detector(image=urban_gray,sigma=1.5,theta_corn=0.005)
visualise(urban,points)


# 2.4 ΠΟΛΥΚΛΙΜΑΚΩΤΗ ΑΝΙΧΝΕΥΣΗ BLOBS

# In[25]:


#2.4.1
def find_all_blobs(image,sigma,theta,scale,N):
  all_sigmas = [(sigma*(scale**i)) for i in range(N)]
  all_points = []
  for s in all_sigmas:
    p = hessian_detector(image=image,sigma=s,theta_corn=theta)
    all_points.append(p)
  return all_points


# In[26]:


def hessian_laplace_detector(image,s0,scale,N,theta):
  all_logs,all_sigmas = find_all_LoG(image=image,s0=s0,scale=scale,N=N)
  all_corners = find_all_blobs(image=image,sigma=s0,scale=scale,N=N,theta=theta)
  result = [] #contains the coordinates of final edges
  for index in range(0,N):
    corner_coord = all_corners[index]
    cur_log = all_logs[index]
    if(index>0 and index<N-1):
      prev_log = all_logs[index-1]
      next_log = all_logs[index+1]
    elif(index==0):
      prev_log = all_logs[index+1]
      next_log = all_logs[index+1]
    elif(index==N-1):
      prev_log = all_logs[index-1]
      next_log = all_logs[index-1]

    for p in corner_coord:
      x = int(p[0])
      y = int(p[1])
      if (cur_log[y][x]>prev_log[y][x]) and (cur_log[y][x]>next_log[y][x]):
        result.append([x,y,all_sigmas[index]])

  return np.array(result)


# In[39]:


res1 = hessian_laplace_detector(image=caravaggio_gray,s0=2,scale=1.5,N=4,theta=0.005)
visualise(caravaggio,res1)


# In[40]:


res2 = hessian_laplace_detector(image=urban_gray,s0=2,scale=1.5,N=4,theta=0.005)
visualise(urban,res2)


# 2.5 ΕΠΙΤΑΧΥΝΣΗ ΜΕ ΧΡΗΣΗ BOX FILTERS ΚΑΙ ΟΛΟΚΛΗΡΩΤΙΚΩΝ ΕΙΚΟΝΩΝ

# In[27]:


def Hessian_aprox(image, sigma, visualize_R = False):
    # Calculate cummulative sum of image
    cumsum = np.cumsum(image, 0, dtype = "float64")
    cumsum = np.cumsum(cumsum, 1, dtype = "float64")

    n = 2*np.ceil(3*sigma)+1
    #Box filters dimensions
    dim1 = int(4*np.floor(n/6)+1)
    dim2 = int(2*np.floor(n/6)+1)
    rect_box = np.asarray([[1]*dim2]*dim1)  #Dxx
    trans_rect_box = np.transpose(rect_box)  #Dyy
    squares = np.asarray([[1]*dim2]*dim2)   #Dxy


    filter_n = 3*dim2
    image_y_size = int(image.shape[0])
    image_x_size = int(image.shape[1])

    # Calculate Lxx,Lyy ,Lxy
    #Αρχικοποιηση
    Lxx = np.asarray([[0]*image_x_size]*image_y_size);
    Lyy = np.asarray([[0]*image_x_size]*image_y_size);
    Lxy = np.asarray([[0]*image_x_size]*image_y_size);

    def calc_box_sum(cumsum, y, x, dimy, dimx):
        return cumsum[y - dimy//2][x - dimx//2] + cumsum[y + dimy//2][x + dimx//2] - cumsum[y - dimy//2][x + dimx//2] - cumsum[y + dimy//2][x - dimx//2]
    #Υπολογισμός παραγώγων
    for i in range(filter_n//2, image_y_size - filter_n//2):
        for j in range(filter_n//2, image_x_size - filter_n//2):
            Lxx[i, j] = calc_box_sum(cumsum, i, j - dim2, dim1, dim2) - 2*calc_box_sum(cumsum, i, j, dim1, dim2) + calc_box_sum(cumsum, i, j + dim2, dim1, dim2)
            Lyy[i][j] = calc_box_sum(cumsum, i - dim2, j, dim2, dim1) - 2*calc_box_sum(cumsum, i, j, dim2, dim1) + calc_box_sum(cumsum, i + dim2, j, dim2, dim1)
            Lxy[i][j] = calc_box_sum(cumsum, i - dim2//2 - 1, j - dim2//2 - 1, dim2, dim2) - calc_box_sum(cumsum, i - dim2//2 - 1, j + dim2//2 + 1, dim2, dim2) \
                        - calc_box_sum(cumsum, i + dim2//2 + 1, j - dim2//2 - 1, dim2, dim2) + calc_box_sum(cumsum, i + dim2//2 + 1, j + dim2//2 + 1, dim2, dim2)

    # Find interest points
    R = Lxx*Lyy - (0.9*Lxy)**2

    # Visualize the criterion
    if visualize_R:
        plt.imshow(R, cmap='gray', vmin=np.amin(R), vmax=np.amax(R))
        plt.title(f"Visualization of R criterion with sigma: {sigma}")
        plt.show()


    # Find local maxima
    B_sq = disk_strel(n)
    Cond1 = ( R==cv2.dilate(R,B_sq) )

    # Find corner condition
    Rmax = np.amax(R)
    Cond2 = R > theta_corn*Rmax

    # Find corners
    intersection = Cond1 & Cond2
    corners = []

    for i in range(0, intersection.shape[0]):
        for j in range(0, intersection.shape[1]):
            if intersection[i][j] == True:
                corners.append([j, i, sigma])

    return corners


# In[42]:


# Visualize points between 2 and 10 as sigma
for s in range(2,10,2):
  corners = Hessian_aprox(caravaggio_gray, s, True)


# In[43]:


# Visualize points between 2 and 10 as sigma
for s in range(2,10,2):
  corners = Hessian_aprox(urban_gray, s, True)


# In[44]:


corners = Hessian_aprox(caravaggio_gray, 2,False)
visualise(caravaggio, corners)
plt.title("Edges detected with single sigma")
plt.show()

corners = Hessian_aprox(urban_gray, 2, False)
visualise(urban, corners)
plt.title("Edges detected with single sigma")
plt.show()


# In[28]:


#2.5.4
# Multiscale box filter
def find_all_box(image,sigma,scale,N):
  all_sigmas = [(sigma*(scale**i)) for i in range(N)]
  all_points = []
  for s in all_sigmas:
    p = Hessian_aprox(image=image,sigma=s)
    all_points.append(p)
  return all_points


# In[29]:


def box_multiscale(image,sigma,scale,N):
    all_logs,all_sigmas = find_all_LoG(image=image,s0=sigma,scale=scale,N=N)
    all_corners = find_all_box(image=image,sigma=sigma,scale=scale,N=N)
    result = []
    for index in range(0,N):
        corner_coord = all_corners[index]
        cur_log = all_logs[index]
        if(index>0 and index<N-1):
              prev_log = all_logs[index-1]
              next_log = all_logs[index+1]
        elif(index==0):
              prev_log = all_logs[index+1]
              next_log = all_logs[index+1]
        elif(index==N-1):
              prev_log = all_logs[index-1]
              next_log = all_logs[index-1]

        for p in corner_coord:
              x = int(p[0])
              y = int(p[1])
              if (cur_log[y][x]>prev_log[y][x]) and (cur_log[y][x]>next_log[y][x]):
                result.append([x,y,all_sigmas[index]])

    return np.array(result)


# In[47]:


corners = box_multiscale(image=caravaggio_gray,sigma=2,scale=1.5,N=4)
visualise(caravaggio, corners)
plt.show()

corners = box_multiscale(image=urban_gray,sigma=2,scale=1.5,N=4)
visualise(urban, corners)
plt.show()


# ### Μέρος 3.Εφαρμογές σε Ταίριασμα και Κατηγοριοποίηση Εικόνων με χρήση Τοπικών Περιγραφητών στα Σημεία Ενδιαφέροντος
# 
# 

# #### 3.1. Ταίριασμα Εικόνων υπό Περιστροφή και Αλλαγή Κλίμακας

# In[30]:


#3.1.1
import cv24_lab1_part3_utils as utils3


#Τοπικοί περιγραφητές που θα χρησιμοποιηθούν HOG,SURF
surf = lambda I,kp: utils3.featuresSURF(I,kp)
hog = lambda I,kp: utils3.featuresHOG(I,kp)

#Aνιχνευτές του part 2 που θα χρησιμοποιθούν
harris = lambda I: harris_stephens_detector(image=I,s=2,p=1.5,k=0.1)
harris_laplace = lambda I: harris_laplace_detector(image=I,s0=1,p0=1.5,scale=1.5,N=4)
hessian = lambda I: hessian_detector(image=I,sigma=1,theta_corn=0.2)
hessian_laplace = lambda I: hessian_laplace_detector(image=I,s0=1,scale=1.5,N=4,theta=0.2)
box = lambda I: box_multiscale(image=I,sigma=1,scale=1.5,N=4)


# In[31]:


from scipy.io import loadmat
file = loadmat("snrImgSet.mat") # reaturns a dictionary
image_set = file['ImgSet'][0] # contains 3 1x20 subsets

bike_set = image_set[0][0] # contains 20 versions of the same bike image
car_set = image_set[1][0]  # contains 20 versions of the same car image
family_set = image_set[2][0] # contains 20 versions of the same family image


# In[47]:


# We used the code given in test_matching_evaluation.py
#SURF
#_________________________________
# calculating using surf
avg_scale_errors, avg_theta_errors = utils3.matching_evaluation(harris, surf)
print("Results for Harris Detector and SURF:")
print(f'Avg. Scale Error for Image 1 : {avg_scale_errors[0]}')
print(f'Avg. Theta Error for Image 1 : {avg_theta_errors[0]}\n')

avg_scale_errors, avg_theta_errors = utils3.matching_evaluation(hessian, surf)
print("Results for Hessian Detector and SURF:")
print(f'Avg. Scale Error for Image 1 : {avg_scale_errors[0]}')
print(f'Avg. Theta Error for Image 1 : {avg_theta_errors[0]}\n')

avg_scale_errors, avg_theta_errors = utils3.matching_evaluation(hessian_laplace, surf)
print("Results for Multiscale Hessian Detector and SURF:")
print(f'Avg. Scale Error for Image 1 : {avg_scale_errors[0]}')
print(f'Avg. Theta Error for Image 1 : {avg_theta_errors[0]}\n')

avg_scale_errors, avg_theta_errors = utils3.matching_evaluation(box, surf)
print("Results for Multiscale Box and SURF:")
print(f'Avg. Scale Error for Image 1 : {avg_scale_errors[0]}')
print(f'Avg. Theta Error for Image 1 : {avg_theta_errors[0]}\n')


# In[34]:


avg_scale_errors, avg_theta_errors = utils3.matching_evaluation(harris_laplace, surf)
print("Results for Harris Multiscale and SURF:")
print(f'Avg. Scale Error for Image 1 : {avg_scale_errors[0]}')
print(f'Avg. Theta Error for Image 1 : {avg_theta_errors[0]}\n')


# In[32]:


# calculating using hog
#HOG
#_______________________
avg_scale_errors, avg_theta_errors = utils3.matching_evaluation(harris, hog)
print("Results for Harris Detector and HOG:")
print(f'Avg. Scale Error for Image 1 : {avg_scale_errors[0]}')
print(f'Avg. Theta Error for Image 1 : {avg_theta_errors[0]}\n')

avg_scale_errors, avg_theta_errors = utils3.matching_evaluation(hessian, hog)
print("Results for Harris Detector and HOG:")
print(f'Avg. Scale Error for Image 1 : {avg_scale_errors[0]}')
print(f'Avg. Theta Error for Image 1 : {avg_theta_errors[0]}\n')

avg_scale_errors, avg_theta_errors = utils3.matching_evaluation(hessian_laplace, hog)
print("Results for Multiscale Hessian Detector and HOG:")
print(f'Avg. Scale Error for Image 1 : {avg_scale_errors[0]}')
print(f'Avg. Theta Error for Image 1 : {avg_theta_errors[0]}\n')

avg_scale_errors, avg_theta_errors = utils3.matching_evaluation(box, hog)
print("Results for Multiscale Box and HOG:")
print(f'Avg. Scale Error for Image 1 : {avg_scale_errors[0]}')
print(f'Avg. Theta Error for Image 1 : {avg_theta_errors[0]}\n')


# In[ ]:


avg_scale_errors, avg_theta_errors = utils3.matching_evaluation(harris_laplace, hog)
print("Results for Harris Multiscale and HOG:")
print(f'Avg. Scale Error for Image 1 : {avg_scale_errors[0]}')
print(f'Avg. Theta Error for Image 1 : {avg_theta_errors[0]}\n')


# # 3.2 Κατηγοροποίηση Εικόνων

# In[33]:


# 3.2.1
# we already have created the wrappers of our detectors
# finding features using surf .... (saving to feats_surf1.txt because takes took long)
feats_surf1 = utils3.FeatureExtraction(harris_laplace,surf, saveFile = "feats_surf1.txt")


# In[78]:


feats_surf2 = utils3.FeatureExtraction(hessian_laplace,surf)


# In[33]:


feats_surf3 = utils3.FeatureExtraction(box,surf)


# In[ ]:


# finding features using hog ... (saving to feats_hog1.txt because takes took long)
feats_hog1 = utils3.FeatureExtraction(harris_laplace,hog, , saveFile = "feats_hog1.txt") 


# In[80]:


feats_hog2 = utils3.FeatureExtraction(hessian_laplace,hog)


# In[83]:


feats_hog3 = utils3.FeatureExtraction(box,hog) 


# In[34]:


# Find accuracy for harris-surf
acc = []
for k in range(5):
    data_train, label_train, data_test, label_test = utils3.createTrainTest(feats_surf1, k)
    BOF_tr, BOF_ts = utils3.BagOfWords(data_train, data_test)
    acc, preds, probas = utils3.svm(BOF_tr, label_train, BOF_ts, label_test)
    accs.append(acc)
print('Mean accuracy for Harris-Laplace with SURF descriptors: {:.3f}%'.format(100.0*np.mean(accs)))



# In[79]:


# Find accuracy for hessian-surf
accs = []
for k in range(5):
    data_train, label_train, data_test, label_test = utils3.createTrainTest(feats_surf2, k)
    BOF_tr, BOF_ts = utils3.BagOfWords(data_train, data_test)
    acc, preds, probas = utils3.svm(BOF_tr, label_train, BOF_ts, label_test)
    accs.append(acc)
print('Mean accuracy for Hessian-Laplace with SURF descriptors: {:.3f}%'.format(100.0*np.mean(accs)))


# In[35]:


# Find accuracy for box-surf
accs = []
for k in range(5):
    data_train, label_train, data_test, label_test = utils3.createTrainTest(feats_surf3, k)
    BOF_tr, BOF_ts = utils3.BagOfWords(data_train, data_test)
    acc, preds, probas = utils3.svm(BOF_tr, label_train, BOF_ts, label_test)
    accs.append(acc)
print('Mean accuracy for Multiscale Box with SURF descriptors: {:.3f}%'.format(100.0*np.mean(accs)))


# In[ ]:


# Find accuracy for harris-hog

acc = []
for k in range(5):
    data_train, label_train, data_test, label_test = utils3.createTrainTest(feats_hog1, k)
    BOF_tr, BOF_ts = utils3.BagOfWords(data_train, data_test)
    acc, preds, probas = utils3.svm(BOF_tr, label_train, BOF_ts, label_test)
    accs.append(acc)
print('Mean accuracy for Harris-Laplace with HOG descriptors: {:.3f}%'.format(100.0*np.mean(accs)))


# In[102]:


# Find accuracy for hessian-hog
accs = []

for k in range(5):
    data_train, label_train, data_test, label_test = utils3.createTrainTest(feats_hog2, k)
    BOF_tr, BOF_ts = utils3.BagOfWords(data_train, data_test)
    acc, preds, probas = utils3.svm(BOF_tr, label_train, BOF_ts, label_test)
    accs.append(acc)
print('Mean accuracy for Hessian-Laplace with HOG descriptors: {:.3f}%'.format(100.0*np.mean(accs)))


# In[104]:


# Find accuracy for box-hog
accs = []
for k in range(5):
    data_train, label_train, data_test, label_test = utils3.createTrainTest(feats_hog3, k)
    BOF_tr, BOF_ts = utils3.BagOfWords(data_train, data_test)
    acc, preds, probas = utils3.svm(BOF_tr, label_train, BOF_ts, label_test)
    accs.append(acc)
print('Mean accuracy for Multiscale Box with HOG descriptors: {:.3f}%'.format(100.0*np.mean(accs)))

