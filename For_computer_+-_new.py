import numpy as np
from mpi4py import MPI
import time

global v
global y
global x
global itog1, itog
global fi_i
global C, C_2, C_4, T, C_2_256, C_2_600


def angle_between(v1,v2):
    dot_pr = v1.dot(v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(dot_pr / norms))
def angle_between1(v1,v2):
    dot_pr = v1.dot(v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    if np.rad2deg(np.arccos(dot_pr / norms))>90:
        return 180-np.rad2deg(np.arccos(dot_pr / norms))
    else:
        return np.rad2deg(np.arccos(dot_pr / norms))
def iter(n):
    global v
    global y
    global x
    global itog1, itog
    global fi_i
    global C, C_2, C_4, T, C_2_256, C_2_600
    k=((v[1])/(v[0]))
    b=y-k*x
    divid=b/k
    divid_600=(512-b)/k
    multi=b*k
    if ((C_2_256)>-divid>256) and (round(y,n)!=0): # проверяем попали ли мы в нейтральный элемент (1)
        y=0
        x=-divid
        v[1]=-v[1]
        if (itog1[0]<x): # проверка на то + или - угол между нормалью и частью траектории
            itog1=np.array([x,y,angle_between1(v,np.array([0,1]))])
            my_file.write(str(x-256)+' '+str(angle_between1(v,np.array([0,1]))) +'\n')
            my_file.flush()

        else:
            itog1=np.array([x,y,-angle_between1(v,np.array([0,1]))])
            my_file.write(str(x-256)+' '+str(-angle_between1(v,np.array([0,1]))) +'\n')
            my_file.flush()


    elif ((C_2_256)>divid_600>256) and (512!=round(y,n)): # аналогично
        y=512
        x=divid_600
        v[1]=-v[1]
        if(itog1[0]<x):
            itog1=np.array([x,y,-angle_between1(v,np.array([0,-1]))])
            my_file.write(str(256+C_2-x+np.pi*256+2*C)+' '+str(-angle_between1(v,np.array([0,-1]))) +'\n')
            my_file.flush()

        else:
            itog1=np.array([x,y,angle_between1(v,np.array([0,-1]))])
            my_file.write(str(256+C_2-x+np.pi*256+C_2)+' '+str(angle_between1(v,np.array([0,-1]))) +'\n')
            my_file.flush()

    else:# вычисления для соударений с выпуклой частью
        d=(4*(256-multi+256*k)**2-4*(1+k**2)*(b-256)**2)
        x1=0
        x2=0
        if (d>=0):
            x1=(2*(256-multi+256*k)+np.sqrt(d))/2/(1+k**2)
            x2=(2*(256-multi+256*k)-np.sqrt(d))/2/(1+k**2)
        else:
            d=-1       
        if (d>=0)and(((0<x1<=256) and (round(x,n)!=round(x1,n))) or ((0<x2<=256) and (round(x,n)!=round(x2,n)))):
                if ((0<x1<=256) and (round(x,n)!=round(x1,n))):
                    x=x1
                    y=k*x+b
                    r=np.array([256-x,256-y])
                    v=v-(2*v.dot(r)/(r.dot(r)))*r
                    if (y<itog1[1]):
                        itog1=np.array([x,y,angle_between1(r,v)])
                        my_file.write(str(C_4+np.pi*256+angle_between(-r,np.array([0,1]))/180*np.pi*256)+' '+str(angle_between1(r,v)) +'\n')
                        my_file.flush()

                    else:
                        itog1=np.array([x,y,-angle_between1(r,v)])
                        my_file.write(str(C_4+np.pi*256+angle_between(-r,np.array([0,1]))/180*np.pi*256)+' '+str(-angle_between1(r,v)) +'\n')
                        my_file.flush()

                else:
                    x=x2
                    y=k*x+b
                    r=np.array([256-x,256-y])
                    v=v-(2*v.dot(r)/(r.dot(r)))*r
                    if (y<itog1[1]):
                        itog1=np.array([x,y,angle_between1(r,v)])
                        my_file.write(str(C_4+np.pi*256+angle_between(-r,np.array([0,1]))/180*np.pi*256)+' '+str(angle_between1(r,v)) +'\n')
                        my_file.flush()

                    else:
                        itog1=np.array([x,y,-angle_between1(r,v)])
                        my_file.write(str(C_4+np.pi*256+angle_between(-r,np.array([0,1]))/180*np.pi*256)+' '+str(-angle_between1(r,v)) +'\n')
                        my_file.flush()

            
        else:
            d=4*(multi-(C_2_256)-256*k)**2-4*(1+k**2)*((C_2_256)**2+b**2-512*b)
            if (d>=0):
                x1=-(2*(multi-(C_2_256)-256*k)+np.sqrt(d))/2/(1+k**2)
                x2=-(2*(multi-(C_2_256)-256*k)-np.sqrt(d))/2/(1+k**2)
            else:
                d=-1
            if (d>=0)and((((C_2_256)<x1<=(C_2_600)) and (round(x,n)!=round(x1,n))) or (((C_2_256)<x2<=(C_2_600)) and (round(x,n)!=round(x2,n)))):
                if (((C_2_256)<x1<=(C_2_600)) and (round(x,n)!=round(x1,n))):
                    x=x1
                    y=k*x+b
                    r=np.array([(C_2_256)-x,256-y])
                    v=v-(2*v.dot(r)/(r.dot(r)))*r
                    if(y<itog1[1]):
                        itog1=np.array([x,y,-angle_between1(r,v)])
                        my_file.write(str(C_2+angle_between(-r,np.array([0,-1]))/180*np.pi*256)+' '+str(-angle_between1(r,v)) +'\n')
                        my_file.flush()

                    else:
                        itog1=np.array([x,y,angle_between1(r,v)])
                        my_file.write(str(C_2+angle_between(-r,np.array([0,-1]))/180*np.pi*256)+' '+str(angle_between1(r,v)) +'\n')
                        my_file.flush()

                else:
                    x=x2
                    y=k*x+b
                    r=np.array([(C_2_256)-x,256-y])
                    v=v-(2*v.dot(r)/(r.dot(r)))*r
                    if(y<itog1[1]):
                        itog1=np.array([x,y,-angle_between1(r,v)])
                        my_file.write(str((C_2)+angle_between(-r,np.array([0,-1]))/180*np.pi*256)+' '+str(-angle_between1(r,v)) +'\n')
                        my_file.flush()

                    else:
                        itog1=np.array([x,y,angle_between1(r,v)])
                        my_file.write(str((C_2)+angle_between(-r,np.array([0,-1]))/180*np.pi*256)+' '+str(angle_between1(r,v)) +'\n')
                        my_file.flush()
            else:
                if (n < 8):
                     iter(n+1)
                     print (1)
                else:
                    my_file.write('11111111111111111111111111111111111111111111111111111\n')
                    my_file.flush()


start=time.time()
collisions=100000 # N (кол-во соударений в одном тесте)
begin=1 # кол-во тестов в одной параллели (в нашем случае 1)
C =16.0 # половина длины нейтрального элемента (L/2) (считая, что для биллиарда бунимовича это значение 256)
C_2=C*2
C_4=C_2*2
T=np.pi*256
T_180=T*180
C_2_256=C_2+256
C_2_600=C_2+512

comm = MPI.COMM_WORLD
rank = 30+comm.Get_rank()
#print('My rank is ',rank) # выводим номер потока
file_name = 'new_testfirststep{}.txt'.format(rank)
my_file = open(file_name, "w") 
itog=np.array([0,0])
itog1=np.array([0,0,0])
v=np.array([1.1,1.1])
#np.random.seed(rank) 



for i in range (begin):
    fi_i=np.random. random()*2*np.pi #задаем случайный угол для старта 
    print ('My rank is ',rank,' ',fi_i)
    x=256+C #задаем координаты центра
    y=256
    #print (x,y,fi_i/2/np.pi*360)
    for j in range (collisions):
        #print (collisions)
        if (j==0): # вычисляем первое соударение
            if (fi_i<np.arctan(256/C)) or (fi_i>=2*np.pi-np.arctan(256/C)): #попадет ли на правую полуокружность, если да, то вычисляем новые x и y
                k=np.tan(fi_i)
                b=y-k*x
                a=1+k**2
                b1=2*k*b-2*(256+2*C)-2*256*k
                c=(256+2*C)**2+b**2-2*256*b
                if ((-(b1)+np.sqrt(b1**2-4*a*c))/2/a>(256+2*C)):
                    x=((-(b1)+np.sqrt(b1**2-4*a*c))/2/a)
                else:
                    x=((-(b1)-np.sqrt(b1**2-4*a*c))/2/a)
                y=x*k+b
                v=np.array([x-(256+C),y-256]) # вектор перемещения
                r=np.array([(256+2*C)-x,256-y]) # вектор перпендикулярный к касательной в т. соударения 
                v=v-(2*v.dot(r)/(r.dot(r)))*r # отражаем вектор
                fi_i=angle_between1(np.array([x-(256+C),y-256]),r) # вычисляем угол от нормали
                itog1=np.array([x,y,fi_i])
                my_file.write(str(C_2+angle_between(-r,np.array([0,-1]))/180*np.pi*256)+' '+str(angle_between1(r,v)) +'\n')
                my_file.flush()
            elif (np.pi-np.arctan(256/C)>fi_i>=np.arctan(256/C)): #далее по аналогии для остальных частей
                x=(256*2-256+np.tan(fi_i)*(256+C))/np.tan(fi_i)
                y=2*256
                v=np.array([x-(256+C),y-256])
                v[1]=-v[1]
                fi_i=angle_between1(np.array([x-(256+C),y-256]),np.array([0,-1]))
                itog1=np.array([x,y,fi_i])
                my_file.write(str(C_2_256-x+np.pi*256+2*C)+' '+str(angle_between1(v,np.array([0,-1]))) +'\n')
                my_file.flush()               
            elif  (np.pi+np.arctan(256/C)<=fi_i<2*np.pi-np.arctan(256/C)):
                x=(-256+np.tan(fi_i)*(256+C))/np.tan(fi_i)
                y=0
                v=np.array([x-(256+C),y-256])
                v[1]=-v[1]
                fi_i=angle_between1(np.array([x-(256+C),y-256]),np.array([0,1]))
                itog1=np.array([x,y,fi_i])
                my_file.write(str(x-256)+' '+str(angle_between1(v,np.array([0,1]))) +'\n')
                my_file.flush()
            else:
                b=256-np.tan(fi_i)*(256+C)
                k=np.tan(fi_i)
                a=1+k**2
                b1=-2*(256-k*b+256*k)
                c=(b-256)**2
                if ((-(b1)+np.sqrt(b1**2-4*a*c))/2/a<=256):
                    x=((-(b1)+np.sqrt(b1**2-4*a*c))/2/a)
                else:
                    x=((-(b1)-np.sqrt(b1**2-4*a*c))/2/a)
                y=x*k+b
                v=np.array([x-(256+C),y-256])
                r=np.array([256-x,256-y])
                v=v-(2*v.dot(r)/(r.dot(r)))*r
                fi_i=angle_between1(np.array([x-(256+C),y-256]),r)
                itog1=np.array([x,y,fi_i])
                my_file.write(str((2*C)*2+np.pi*256+angle_between(-r,np.array([0,1]))/180*np.pi*256)+' '+str(angle_between1(r,v)) +'\n')
                my_file.flush()
        else:
            iter (6)
        
end=time.time()  
print (end-start) 
my_file.close()



        




