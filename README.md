# TMS

**Transporation Management System**

경로에 영향을 줄 수 있는 변수를 사용자가 GUI를 통해 조절할 수 있도록 하여 조절값에 따라 경로가 변경되어질 수 있도록 설계했습니다.
사용된 알고리즘은 메타휴리스틱 알고리즘의 LS와 GA입니다.


# 알고리즘
`cost 개념`을 사용해 사용자가 설정한 기준에서 멀어질 수록 비용이 추가됩니다. 그 비용이 가장 적은 곳으로 먼저 가도록 구현했습니다.  
업로드 해놓은 이미지를 보면 검은 원은 `허브의 반경`을 뜻합니다. 이 반경은 허브로부터 일정 거리를 기준으로 만들었습니다.  
각 차량의 색을 다르게 하였습니다.  
빨간색 숫자로 표시되는 것은 지점의 ID라고 보시면 됩니다.  


# 파일 설명
**Local Search** 와 **Genetic Algorithm**으로 나누어져 있습니다.  
  
파일 내부는 `TMS`, `Parser`, `ui`, `Heuristic`, `Component` 로 구성되어 있습니다.  
   
- `TMS`는 실행파일입니다. GUI에서 설정한 변수 값을 이용해서 실행 됩니다.
  각 차량을 순차적으로 뽑아 진행할지, 아니면 동시에 후보군에 넣어서 가장 적은 비용을 가지는 차량을 선택하면서 진행할지를 구분했습니다. 그렇게 하다보니
  동시진행과 순차진행 버전이 생겼습니다만... 다 구현을 하고 나니 순차진행 버전은 동시진행 버전에 비해 크게 메리트가 없었지만 기록을 위해 남겨놓았습니다.  
- `Parser` 엑셀 파일로 만들어진 데이터를 읽고 정의해주는 파일입니다.  
- `ui` 는 PyQt로 만든 간단한 GUI 입니다. 여기에서 변수 값을 조절할 수 있습니다.  
- `Heuristic` 메타 휴리스틱 알고리즘을 사용하는 **메인**파일이라고 볼 수 있습니다.  
  ### ***code 파일에 주석으로 함수나 코드에 대한 부분적인 기록을 해놨지만 밑에서 조금 더 자세하게 알려드리겠습니다.***  
- `Component` 파일은 차량과 물품 등의 클래스를 정의합니다.  

# *Heuristic*에 대해서
전체적인 코드의 흐름은 `비용` 개념을 따라갑니다. 매 반복마다 가야하는 지점들에 대한 비용을 계산합니다.  
  
예를 들어 1,2,3 차량이 존재하고 A, B 지점을 가야합니다. 이 상황에서 각 차량은 클래스 객체이므로 cost_list라는 속성을 가집니다.  
  
cost_list에는 각 차량이 A, B지점을 가는 비용을 경로가 교차되었는지, 시간이 초과되었는지 등 각종 비용함수들을 통해 계산합니다.  
  
계산한 결과를 바탕으로 A지점을 가는 비용이 가장 적은 2번 차량, B지점을 가는 비용이 가장 적은 1번 차량 이런 식으로 선택됩니다.  
  
또한 차량이 가지는 cost 속성으로 현재까지의 cost를 계산합니다.  
  
조금 더 효율적인 차량 배치를 위해 Polygon을 이용합니다. 각 지점들을 연결하여 전체 Polygon을 만들고 허브를 기준으로 하여 구역을 나눕니다.  
  
각 구역을 갈 수 있는 차량을 지정하여 다른 구역에 갈 경우 큰 비용을 부과하여 가지 못하도록 만듭니다.


# 실행방법

TMS_1 을 실행하면 GUI가 뜨는데 그곳에서 지역과 값을 조절할 수 있습니다.
업로드한 이미지의 결과처럼 뜨기 위해서는
400, 3000, 7000, 0.95, 2, 2, 4000, 1000, 1000, 80, 2, 0.9, 0.9 (위에서 아래 순서) 로 설정하시면 됩니다.
필수 선택에서 순차 진행 혹은 동시 진행 1개만 선택해주세요.

# <값 설명>
`과적 금지`: 과적을 하지 않습니다.  
  
`초과 근무 금지`: 초과 근무를 하지 않습니다.  
  
`물량 비용`: 차가 가지고 있는 물량이 많을수록 물량 비용이 추가되어 적절한 물량을 유지할 수 있도록 합니다.  
  
`반경 비용`: 허브(hub)에서 일정 거리의 반경을 생성했습니다. 그 반경을 넘어가면 추가되는 비용입니다. 허브와의 적절한 거리 유지와 먼 거리를 가지 않도록 하기 위함입니다.
  
`교차 비용`: 경로가 생성될 때 불필요한 교차가 생기지 않도록 계산합니다.  
  
`최대 적재`: 과적 금지와는 다른 값으로 과적 금지의 여부에 상관 없이 최대 적재할 수 있는 양을 설정합니다. 0.8이면 80%입니다.  
  
`제한 용량`: 과적 금지의 여부에 상관 없이 최대 적재 용량에서 +x kg 까지 가능하도록 합니다.  
  
`최대 시간`: 초과 근무 금지의 여부에 상관 없이 원래 정해져 있는 근무 시간에서 +x 시간까지 근무 할 수 있도록 합니다.  
  
`차와 지점간의 거리 비용`: 현재 차가 위치 하고 있는 곳과 다음 지점간의 거리가 멀면 추가되는 비용입니다. 가까운 지점으로 먼저 갈 수 있도록 하기 위함입니다.  
  
`최대 중량 비용`: 최대 적재량으로부터 설정한 제한 용량까지 적재한 중량 비용을 부과합니다.  
  
`시간 초과 비용`: 시간을 초과하면 추가되는 비용입니다.  
  
`차량 속도`: 차량 속도를 정합니다.  
  
`상하차 시간`: 상하차 시간을 임의로 정합니다.  
  

<동시 진행만 해당> - 차량 제외 조건  
동시 진행의 경우에는 모든 차들이 예비 선상에 올라와 어떤 차가 가는 게 가장 효율적인지를 계산합니다.  
경로를 생성하면서 차량의 적재량이나 시간이 다 되었을 때 차량을 제외시킵니다.  
차량 제외를 하기 위한 조건입니다.  
  
`사용된 근무 시간의 X%를 충족하면 차량을 제외` : 예를 들어 근무시간이 9시간이라면 9시간을 꽉 채울 수도 있으나 80%만 사용되도 차량을 제외할 수 있습니다.  
  
`사용된 차의 용량의 X%를 충족하면 차량을 제외` : 정해진 차의 용량의 X%를 충족하면 차량을 제외시킵니다.  

