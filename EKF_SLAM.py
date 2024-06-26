from numpy import array, sqrt
import seaborn as sns
import numpy as np
from math import sqrt, tan, cos, sin, atan2
import matplotlib.pyplot as plt
import random

# Выбрать, что отображать в качестве второго графика
CovariationMatrix = 0
KalmanGain = 1
KalmanGainMatrix = 2

choise = CovariationMatrix

lastLandmark = 1
landmarks = []

# Расширенный фильтр Калмана для модели колесного робота с 4-мя колесами и рулевым управлением
# Управление осуществляется скоростью и углом поворота передних колес
# Измерения местоположения осуществляются по ориентирам, которые определяются расстоянием и углом до каждого
class EKFslam():
    def __init__(self, dt, std_r, std_phi, wheelbase):
        self.dt = dt
        self.wheelbase = wheelbase
        # self.std_vel = std_vel
        # self.std_steer = std_steer
        self.std_r = std_r
        self.std_phi = std_phi
        self.X = np.zeros((3,1)) # [x, y, theta]
        self.P = np.zeros((3,3)) # Px - матрица ошибки ковариации для вектора состояния, но пока без ориентиров
        self.NofObs = 0          # Количество обнаруженных ориентиров
        self.R = np.array([[1,0,0],
                           [0,1,0], # Ковариация шума процесса
                           [0,0,0.1]])

    # Добавляет единичную матрицу 2х2 в правый нижний угол к матрице P
    # Нужно для добавления ошибки ковариации ориентира в матрицу ковариации P
    def matrixPincrease(self, P):
        size = int(sqrt(np.size(P)))
        g = np.zeros((2, size))
        I = np.array([[99999999,0],
                      [0,99999999]])
        g1 = np.zeros((size, 2))
        I0 = np.concatenate((g1, I), axis=0)
        a1 = np.concatenate((P, g), axis=0)
        P1 = np.concatenate((a1, I0), axis=1)
        return P1

    # Необходимо для исправления ошибки в уравнении X = X_ + K(z-h(X_))
    # Если в z угол будет к примеру 1 градус, а в h(x) 359 градусов, то
    # Разница между ними 2 градуса, но алгоритм посчитает, что -358 и испортит все измерения
    def residual(self, a, b):
        # """ compute residual (a-b) between measurements containing
        # [range, bearing]. Bearing is normalized to [-pi, pi)"""
        y = a - b
        y[1] = y[1] % (2 * np.pi)  # force in range [0, 2 pi)
        if y[1] > np.pi:  # move to [-pi, pi)
            y[1] -= 2 * np.pi
        return y

    # Этап предсказания фильтра - обновляет матрицу ковариации и вектор состояния
    def predict(self, u):
        #x_t=f(x_t-1, ut)
        v = u[0]
        steering_angle = u[1]
        theta = self.X[2]
        r = self.wheelbase / tan(steering_angle)
        d = v * self.dt
        beta = (d/self.wheelbase)*tan(steering_angle)
        a = -r * np.sin(theta) + r * np.sin(theta + beta)
        b = r * np.cos(theta) - r * np.cos(theta + beta)
        Xadd = np.array([[a.item(),b.item(),beta.item()]])
        #print(np.size(Xadd))
        # print('Xadd = \n',Xadd)
        g = np.zeros((3, self.NofObs * 2))
        I = np.eye(3)
        F1 = np.concatenate((I, g), axis=1)
        # print(F1)
        self.X = self.X + np.dot(np.transpose(F1),np.transpose(Xadd))
        # print('predict: X_ = \n',self.X,'\n----')

        #P_t = F * Pt-1 * F^T + F1^T * Rt * F1
        Fj = np.array([[0,0,a.item()],
                       [0,0,b.item()],
                       [0,0,   0    ]])
        F1TFj = np.dot(np.transpose(F1),Fj)
        I = np.eye(3+self.NofObs * 2)
        F = I + np.dot(F1TFj,F1)
        #print('F =\n', F)
        FdotP = np.dot(F,self.P)
        FPFT = np.dot(FdotP,np.transpose(F))
        #print(FPFT)
        FTdotR = np.dot(np.transpose(F1),self.R)

        Rt = np.dot(FTdotR,F1)
        # print(Rt)
        self.P = FPFT + Rt
        # print('predict: P = \n', self.P, '\n----')

    # Этап корректировки фильтра
    def update(self, z):
        for landmark in z:
            r = landmark[1]
            phi = landmark[2]
            # Если ориентир еще не был обнаружен, то добавить его оценку в вектор состояний
            if landmark[0] > self.NofObs:
                # Вычисление координат ориентира по оценке местоположения робота и измерениям
                # print('landmark = ', landmark)
                theta = self.X[2]
                theta = theta.item()
                # print('theta = ', theta)
                Xf1 = self.X[0] + r * np.cos(phi + theta)
                Xf2 = self.X[1] + r * np.sin(phi + theta)
                Xf = np.array([[Xf1.item(),Xf2.item()]])
                # print('Xf = ', Xf)
                # Добавляем координаты ориентира в вектор состояния
                self.X = np.concatenate((self.X,np.transpose(Xf)))
                # print('X = \n',self.X,'\n---------------------')
                # Увеличиваем матрицу ковариации для нового измерения
                self.P = self.matrixPincrease(self.P)
                self.NofObs+=1

            # Формируем вектор z_
            # Достаем элемент вектора 3+2*n-1 соответствующий px (2 вместо 3, потому что индексация с нуля)
            px = self.X[2+2*landmark[0]-1]
            py = self.X[2+2*landmark[0]]
            betaX = px-self.X[0]
            betaX = betaX.item()
            betaY = py-self.X[1]
            betaY = betaY.item()
            beta = np.array([[betaX,betaY]])
            # print('beta = ', beta)
            q = np.dot(beta,np.transpose(beta))
            q = q.item()
            theta = self.X[2]
            theta = theta.item()
            z_ = np.array([[sqrt(q),np.arctan2(betaY,betaX)-theta]])
            z_ = np.transpose(z_)
            # print('z_ = ', z_)

            # Вычисление вектора H
            I = np.eye(3)
            Fx = np.concatenate((I,np.zeros((2,3))))
            Fx = np.concatenate((Fx,np.zeros((5,self.NofObs*2))),axis=1)
            Fx[3][2+landmark[0]*2-1]=1
            Fx[4][2 + landmark[0] * 2] = 1
            #print('Fx =\n',Fx)
            H = np.array([[-sqrt(q)*betaX, -sqrt(q)*betaY, 0, sqrt(q)*betaX, sqrt(q)*betaY],
                          [betaY, -betaX, -q, -betaY, betaX]])
            H = (1/q)*np.dot(H,Fx)
            # print('H = \n',H)

            # Вычисление коэффициента Калмана K
            HdotP = np.dot(H,self.P)
            # print('HdotP = \n', HdotP)
            HPHT = np.dot(HdotP,np.transpose(H))
            Qt = np.array([[self.std_r**2,0],
                           [0,self.std_phi**2]])
            #print('Qt = \n', Qt)
            PHT = np.dot(self.P,np.transpose(H))
            #print('PHT = \n', PHT)
            K = np.dot(PHT,np.linalg.inv(HPHT+Qt))

            self.K = K
            #print('K = \n', K)

            # Вычисление вектора измерения z
            z = np.array([[r,phi]])
            z = np.transpose(z)
            #print(z)
            z_z_ = self.residual(z, z_)
            self.X = self.X + np.dot(K,z_z_)
            # print('X = \n',self.X,'\n---------------------')

            # Обновление матрицы ошибки ковариации P
            I = np.eye(int(np.size(K)/2))
            I_KH = I - np.dot(K,H)
            self.P = np.dot(I_KH,self.P)
        #print('X = \n', self.X, '\n---------------------')
        # print('diff between X and landmarks')
        # global landmarks
        # for l in landmarks:
        #     if l[0] != 0:
        #         print('Xx[', l[0] ,'] = ', self.X[2+2*l[0]-1], ': landx = ', l[1], 'diffx = ', l[1] - self.X[2+2*l[0]-1])
        #         print('Xy[', l[0], '] = ', self.X[2 + 2 * l[0]], ': landy = ', l[2], 'diffy = ', l[2] - self.X[2 + 2 * l[0]])
        # print('P = \n', self.P, '\n-------------------')
        return self.X, self.P


# Функция движения робота. Перемещает робота по карте
# Принимает на вход координаты робота и управляющее воздействие
# Управляющее воздействие это скорость робота и угол поворота передних колес
# Возвращает Новые координаты и угол робота после перемещения
def move(wheelbase, x, u, dt):
    hdg = x[2]
    vel = u[0]
    steering_angle = u[1]
    dist = vel * dt

    if abs(steering_angle) > 0.001:  # is robot turning?
        beta = (dist / wheelbase) * tan(steering_angle)
        r = wheelbase / tan(steering_angle)  # radius
        dx = np.array([-r * sin(hdg) + r * sin(hdg + beta),
                       r * cos(hdg) - r * cos(hdg + beta),
                       beta])
    else:  # moving in straight line
        dx = np.array([dist * cos(hdg),
                   dist * sin(hdg),
                   0])

    return x + dx

# Выдает измерения в виде номера ориентира, дальности до него и угла до него с добавлением ошибки
def z_landmark(lmark, sim_pos, std_rng, std_brg):
    x, y, phi = sim_pos[0], sim_pos[1], sim_pos[2]
    px, py = lmark[1], lmark[2]
    numofLandmark = lmark[0]
    d = sqrt((px - x)**2 + (py - y)**2)
    a = atan2(py - y, px - x) - phi

    a = a % (2 * np.pi)  # force in range [0, 2 pi)
    if a > np.pi:  # move to [-pi, pi)
        a -= 2 * np.pi

    z = [numofLandmark,
    d + np.random.randn()*std_rng,
    a + np.random.randn()*std_brg]

    return z

# Вычисляет угол до ориентира в радианах. Нужно для оценки видимости ориентира
def angle_to_landmark(robotpose, landmarkpose):
    x, y = robotpose[0], robotpose[1]
    px, py = landmarkpose[1], landmarkpose[2]
    return atan2(py - y, px - x)

# Выдает вектор измерения в виде списка ориентиров [номер ориентира, дальность до ориентира, угол до ориентира]
# С добавлением ошибок
# Назначает ориентирам номера. Номер за ориентиром закрепляется навсегда, не меняется и не повторяется
def Get_Z_from_Landmarks(sim_pos, std_rng, std_brg):
    global landmarks
    global lastLandmark
    z = []
    phi = sim_pos[2]
    for i in range(1,len(landmarks)):
        #print('landmarks[',i,'] = ', landmarks[i])
        # Видимый ли ориентир?
        if angle_to_landmark(sim_pos, landmarks[i]) < phi + 1.05 and angle_to_landmark(sim_pos, landmarks[i]) > phi - 1.05:
            #print('visible landmark')
            # Если ориентир еще не был обнаружен - назначить ему номер
            if landmarks[i][0] == 0:
                landmarks[i][0] = lastLandmark
                lastLandmark +=1
            # Получает измерения для ориентира с добавлением ошибки
            z.append(z_landmark(landmarks[i],sim_pos, std_rng, std_brg))
    return z


# Генерирует массив ориентиров, каждый из которых задается в виде: [обнаружен ли ориентир 0 - нет 1 - да, координата х, координата у]
def Generate_Landmarks(num=50):
    global landmarks
    for i in range(1,num):
        landmarks.append([0,np.random.uniform(0,10),np.random.uniform(-3,6)])
    return landmarks

# SLAM - тип фильтра или алгоритма
# dt - временной интервал дискретизации в секундах
# step - количество интервалов дискретизации, которое ждет фильтр, чтобы произвести оценку положения робота
# std_r - стандартное отклонение дальности до ориентира - ошибка в определении этой дальности то есть
# std_brg - ошибка в определении угла до ориентира
# wheelbase - колесная база робота, нужна в уравнении кинематики робота
# timeEnd - время окончания симуляции в секундах
# std_vel - ошибка в определении скорости робота
# std_steer - ошибка в угле поворота робота
def Simulation(SLAM, dt=0.1, step = 10, std_r=0.01, std_brg=0.001, wheelbase=0.05, timeEnd = 40, std_vel = 0.01, std_steer = 0.02):
    # Инициализация заданного фильтра
    filter = SLAM(dt*step, std_r, std_brg, wheelbase)

    if choise == CovariationMatrix:
        Correlationmatrix = plt.figure('Correlation Matrix P')
    elif choise == KalmanGain:
        Correlationmatrix = plt.figure('Kalman Gain')
    elif choise == KalmanGainMatrix:
        Correlationmatrix = plt.figure('Kalman Gain Matrix')

    figmapandtrack = plt.figure('Map and track')
    # Инициализация массивов для построения графика положений робота
    trackX = []
    trackY = []
    estimatesX = []
    estimatesY = []
    MeasurementsXfromL = []
    MeasurementsYfromL = []
    measurements = []
    # Инициализация начальной позиции робота (всегда в начале координат)
    x = np.array([0, 0, 0])
    # Инициализация управляющего вектора Скорость, угол
    u = [0.5, 0.05]

    landmarks = Generate_Landmarks()

    for t in range(1,1+int(timeEnd/dt)):
        # Изменение управляющего воздействия, чтобы задать путь роботу
        if t == 20:
            u = [1, -0.02]
        if t == 50:
            u = [0.3, 0.05]
        if t == 55:
            u = [0.6, -0.05]
        if t == 140:
            u = [1, -0.01]
        if t == 160:
            u = [0.5, -0.05]
        if t == 200:
            u = [0.5, 0.02]

        x = move(wheelbase, x, u, dt)
        # print('t[',t,'] Pose = ', x)

        # Добавить позиции робота в массив положений робота
        trackX.append(x[0])
        trackY.append(x[1])

        if t % step == 0:
            plt.figure(figmapandtrack)
            plt.scatter(x[0], x[1], marker='s', color='black')

            # Генерируем управляющее воздействие с ошибкой
            u1 = array([u[0] + np.random.randn()*std_vel, u[1] + np.random.randn()*std_steer])
            # print('Pose = ', x)

            # Предсказание
            filter.predict(u=u1)

            z = Get_Z_from_Landmarks(x, std_r, std_brg)

            # Корректировка
            X,P = filter.update(z=z)
            K = filter.K

            # Построение первой матрицы ковариации
            if t == step:
                plt.figure(Correlationmatrix)
                if choise == KalmanGain:
                    KG = []
                    KGtime = []
                    KG.append(np.var(K))
                    KGtime.append(t*dt)

                elif choise == KalmanGainMatrix:
                    tstr = 'time ' + str(t)
                    plt.subplot(3, 3, 1)
                    plt.title(tstr)
                    sns.heatmap(K, cmap="YlOrBr", vmax=1)

                elif choise == CovariationMatrix:
                    tstr = 'time ' + str(t)
                    plt.subplot(3, 3, 1)
                    plt.title(tstr)
                    sns.heatmap(P, cmap="YlOrBr", vmax=1)



            plt.figure(figmapandtrack)
            plt.scatter(X[0], X[1], marker='o', color = 'green')
            estimatesX.append(X[0])
            estimatesY.append(X[1])

        # Построение матриц ковариаций ровно 8 раз в течение всей симуляции с равными интервалами
        if t % (int(timeEnd / dt) // 8) == 0 and t >= (int(timeEnd / dt) // 8):
            plt.figure(Correlationmatrix)

            if choise == KalmanGain:
                KG.append(np.var(K))
                KGtime.append(t * dt)

            elif choise == KalmanGainMatrix:
                tstr = 'time ' + str(t)
                plt.subplot(3, 3, t // (int(timeEnd / dt) // 8)+1)
                plt.title(tstr)
                sns.heatmap(K, cmap="YlOrBr", vmax=1)

            elif choise == CovariationMatrix:
                tstr = 'time ' + str(t)
                plt.subplot(3, 3, t // (int(timeEnd / dt) // 8)+1)
                plt.title(tstr)
                sns.heatmap(P, cmap="YlOrBr", vmax=1)

    if choise == KalmanGain:
        plt.figure(Correlationmatrix)
        plt.plot(KGtime,KG)

    plt.figure(figmapandtrack)
    plt.plot(trackX, trackY, color='k', lw=2, label='Истинные координаты')
    plt.plot(estimatesX, estimatesY, color='g', lw=2, label='EKF SLAM')

    for lmark in landmarks:
        plt.scatter(lmark[1], lmark[2], marker='*', color='r', lw=2)

    plt.legend()
    plt.axis('equal')
    plt.show()



Simulation(EKFslam)



