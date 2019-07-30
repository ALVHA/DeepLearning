
# coding: utf-8

# ## 오차 역전파
# * Gradient를 출력층부터 계산한 후 입력층까지 거꾸로 계산하는 방법
# * 초기 가중치, weight 값은 랜덤으로 주어짐
# * 각각의 노드는 하나의 퍼셉트론으로 생각
# * 노드를 지나칠 떄마다 활성화 함수를 적용
# * 활성화 함수는 시그모이드 함수 사용
# * 결론은 오차를 줄이는 방법
# 

# ## 신경망의 문제
# 
# * 신경망의 첫 문제 :XOR
# * Vanishing Gradient
# * 기본저긍로 Activation 에 ReLu나 Tanh 함수를 ㅎ사용
# * Vanishing Gradient 는 Activation 변경을 통해 어느 정도 보완 가능

# In[9]:

import numpy as np

w11 = np.array([-2, -2])
w12  =np.array([2,2])
w2 = np.array([1,1])

b1 = 3
b2 = -1
b3 = -1


# In[10]:

# 퍼셉트론

def MLP(x, w, b):
    y = np.sum(w*x) + b
    if y <= 0:
        return 0
    else:
        return 1


# In[15]:

def NAND(x1, x2):
    return MLP(np.array([x1, x2]), w11, b1)


# In[16]:

def OR(x1, x2):
    return MLP(np.array([x1, x2]), w12, b2)


# In[17]:

def AND(x1, x2):
    return MLP(np.array([x1, x2]), w2, b3)


# In[18]:

def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))

if __name__ == '__main__':
    
    for x in [(0,0), (1,0), (0,1), (1,1)]:
        y = XOR(x[0], x[1])
        print("입력 값: " +str(x) + "출력 값:"+ str(y))


# In[19]:

import random
import numpy as np


# In[20]:

random.seed(777)

# 환경 변수 지정

# 입력값 및 타겟값
data = [                  ### XOR 데이터로 처음 시작
    [[0, 0], [0]],        ### NAND, OR, AND로도 실행해 볼 것
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]]
]


# In[21]:

# 실행 횟수(iterations), 학습률(lr), 모멘텀 계수(mo) 설정
iterations=5000
lr=0.1
mo=0.9


# In[29]:

## 시그모이드 정의
def sigmoid(x, derivative=False):
    if (derivative==True):
        return x * (1-x)
    return 1 / (1+np.exp(-x))

#tanh 정의
def tanh(x, derivative = False):
    if (derivative == True):
        return 1- x**2
    return np.tanh(x)

def makeMatrix(i, j, fill=0.0):
    mat = []
    for i in range(i):
        mat.append([fill] * j)
    return mat


# In[34]:

## 신경망의 실행
class NeuralNetwork:
    
    def __init__(self, num_x, num_yh, num_yo, bias = 1):
        
        ## 입력값 num_x, 은닉층 num_yh, 출력층 num_yo ,바이어스 1
        self.num_x = num_x + bias ## 바이어스는 1로 지정함
        self.num_yh = num_yh
        self.num_yo = num_yo
        
        self.activation_input = [1.0] * self.num_x
        self.activation_hidden = [1.0] * self.num_yh
        self.activation_out = [1.0] * self.num_yo
        
        # 가중치 입력 초깃값
        self.weight_in = makeMatrix(self.num_x, self.num_yh)
        for i in range(self.num_x):
            for j in range(self.num_yh):
                self.weight_in[i][j] =random.random()
        
        # 가중치 출력 초깃값
        self.weight_out = makeMatrix(self.num_yh, self.num_yo)
        for j in range(self.num_yh):
            for k in range(self.num_yo):
                self.weight_out[j][k] = random.random()
                
        # 모멘텀 SGD를 위한 이전 가중치 초깃값
        self.gradient_in = makeMatrix(self.num_x, self.num_yh)
        self.gradient_out = makeMatrix(self.num_yh, self.num_yo)
        
    def update(self, inputs):
        
        # 입력 레이어의 활성화 함수
        for i in range(self.num_x - 1):
            self.activation_input[i] = inputs[i]
            
        ## 은닉층의 활성화 함수
        for j in range(self.num_yh):
            sum = 0.0
            for i in range(self.num_x):
                sum = sum + self.activation_input[i] *self.weight_in[i][j]
                
                ## 시그모이드와 tanh 중에서 활성화 함수 선택
                self.activation_hidden[j] = tanh(sum, False)
                
        # 출력층의 활성화 함수
        for k in range(self.num_yo):
            sum = 0.0
            for j in range(self.num_yh):
                sum = sum+self.activation_hidden[j] *self.weight_out[j][k]
            
            # 시그모이드와 tanh 중에서 활성화 함수 선택
            self.activation_out[k] = tanh(sum, False)
        
        return self.activation_out[:]
    
    # 역전파의 실행
    def backPropagate(self, targets):
        
        # 델타 출력 계산
        output_deltas = [0.0] * self.num_yo
        for k in range(self.num_yo):
            error = targets[k] - self.activation_out[k]
            ## 시그모이드와 tanh중에서 활성화 함수 선택
            output_deltas[k] = tanh(self.activation_out[k], True) *error
            
            
        # 은닉 노드의 오차 함수
        hidden_deltas = [0.0] *self.num_yh
        for j in range(self.num_yh):
            error = 0.0
            for k in range(self.num_yo):
                error = error + output_deltas[k] * self.weight_out[j][k]
                    
            hidden_deltas[j] = tanh(self.activation_hidden[j] ,True) *error
            
        # 출력 가중치 업데이터
        for j in range(self.num_yh):
            for k in range(self.num_yo):
                gradient = output_deltas[k] * self.activation_hidden[j]
                v = mo * self.gradient_in[j][k] - lr *gradient
                self.weight_in[j][k] += v
                self.gradient_out[j][k] = gradient
                
        # 입력 가중치 업데이트
        for i in range(self.num_x):
            for j in range(self.num_yh):
                gradient = hidden_deltas[j] * self.activation_input[i]
                v = mo*self.gradient_in[i][j] - lr * gradient
                self.weight_in[i][j] += v
                self.gradient_in[i][j] = gradient
                
        # 오차의 계산(최소 제곱법)
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k] - self.activation_out[k])**2
        return error
    
    def train(self, patterns):
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets)
            if i % 500 == 0:
                print('error: %-.5f' % error)
                            
    # 결괏값 출력
    def result(self, patterns):
        for p in patterns:
            print('Input: %s, Predict: %s' % (p[0], self.update(p[0])))


# In[35]:

if __name__ == '__main__':
    
    n = NeuralNetwork(2, 2, 1)
    
    n.train(data)
    
    n.result(data)


# ## 

# In[ ]:



