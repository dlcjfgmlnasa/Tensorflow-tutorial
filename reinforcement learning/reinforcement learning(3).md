# Q-Learning

Q-learning 이랑 action-value function 값을 업데이트 해주면서 가장 나은 action을 취해주는 알고리즘입니다. 오늘은 Frozenlake 문제를 Q-learning에 한번 적용해 볼텐대 지난 시간에 배웠던 action, statue, env의 개념들을 잘 기억하고 계시면 좋을 듯합니다.

간단하게 FrozenLake Environment에 대해 설명드리겠습니다.

agent, 즉 action을 당할 놈은 초록색에서 시작을 해서 저기있는 노란색 위치까지 가야됩니다. 중간에 파란색 부분은 빠지면 게임이 끝나버리는 구멍이라고 보면 됩니다. Environment에서는 action값을 입력해주면 그에 맞는 Next_state, reward, done, info를 리턴해주고 여기서 Next_state는 action을 통해 움직여서 바뀐 state라고 보면 됩니다. reward는 그 action을 했을때 주는 보상값이고 done은 이 환경이 끝났는지 안끝났는지를 알려주는 boolean 값입니다. info는 기타 추가 정보를 가져와줍니다. 여기서 done이 일어날 state는 각각의 파란색 구멍과 도착지점인 노란색 부분이고 이 Env에서는 목표 지점이 아닌 곳으로 움직이면 reward를 0을 주고 목표지점에 도착하면 reward 1 을 줍니다.

![frozen Lake WorldS](https://jaehwant.github.io/assets/images/DRL_01_00.png)

OpenAI Gym 에 있는 FrozenLake-v0 의 Environment 이고 agent가 있을 수 있는 state가 총 16개 밖에 되지 않습니다 action 또한 4개밖에 되지 않기 때문에 Q-table을 이용해도 정상적으로 학습이 진행 될것 입니다.

```python
import gym
import numpy as np

env = gym.make('FrozenLake-v0')

# 모든 가능한 상태(observation_space)와 행동(action_space)에 대해 표의 모든 값을 0으로 초기화한다.
Q = np.zeros([env.observation_space.n,env.action_space.n])
# 학습 파라미터를 설정한다
# lr 은 학습률
lr = .85
# y은 gamma, 할인률 미래 보상에 대해 얼마나 할인할 것인가
y = .99
# 에피소드 수 2000번 수행
num_episodes = 2000
# 에피소드의 각 걸음(step)과 총 보상을 저장하려는 리스트를 만든다
#jList = []
# 에피소드의 보상을 모음
rList = []
for i in range(num_episodes):
    # 첫 상태를 초기화한다 env.reset()
    s = env.reset()
    # 총 보상 rAll
    rAll = 0
    # 끝났는지를 나타내는 d 변수
    d = False
    # 걸음 수는 j
    j = 0
    # Q 러닝 알고리즘
    # 99걸음까지만 허용함
    while j < 99:
        # 걸음마다 더해줌
        j+=1
        # Q table에서 e -greedy 에 따라 가장 좋은 행동을 선택함 매 걸음마다 랜덤적 요소를 넣음
        # 1/ (i+1) 을 넣는 이유는 에피소드가 진행될 수록 랜덤적 요소를 줄이려고 하는 것임
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        # env.step은 주어진 행동에 대한 다음 상태와 보상, 끝났는지 여부, 추가정보를 제공함
        s1,r,d,_ = env.step(a)
        #새로 얻은 보상을 바탕으로 이전의 Q table을 업데이트함
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        # 에피소드 총 보상에서 더해줌
        rAll += r
        # 상태를 다음 상태로 바꿈
        s = s1
        # 끝에 도달하면 다음 에피소드로 넘어감
        if d == True:
            break
    #jList.append(j)
    # 에피소드별 총 보상을 모음
    rList.append(rAll)

print ("Score over time: " +  str(sum(rList)/num_episodes))
print ("Final Q-Table Values")
print (np.round(Q,3))

'''
결과는 62%정도가 됩니다.
Final Q-Table Values
[[ 0.771  0.015  0.015  0.015]
 [ 0.002  0.002  0.     0.589]
 [ 0.     0.339  0.003  0.002]
 [ 0.002  0.     0.002  0.411]
 [ 0.828  0.005  0.002  0.   ]
 [ 0.     0.     0.     0.   ]
 [ 0.     0.     0.07   0.   ]
 [ 0.     0.     0.     0.   ]
 [ 0.006  0.006  0.     0.916]
 [ 0.006  0.945  0.     0.   ]
 [ 0.982  0.     0.003  0.001]
 [ 0.     0.     0.     0.   ]
 [ 0.     0.     0.     0.   ]
 [ 0.     0.     0.963  0.   ]
 [ 0.     0.995  0.     0.   ]
 [ 0.     0.     0.     0.   ]]
'''
```

지금까지 간단한 Q-Table를 이용해서 Q-learning을 진행하는 것을 보실수가 있습니다. 복잡한 연산없이 Q-table를 이용해서 학습이 진행되는데 다음시간에는 Q-Table를 이용한 Q-learning의 문제점과 해결방안에 대해서 설명드리도록하겠습니다.