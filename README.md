# Quantum AI: 핸즈온 양자 머신러닝 with PennyLane
#### 양자 신경망·양자커널·하이브리드 모델 구현 PennyLane 핵심 가이드 

<img src="https://beat-by-wire.gitbook.io/beat-by-wire/~gitbook/image?url=https%3A%2F%2F3055094660-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FYzxz4QeW9UTrhrpWwKiQ%252Fuploads%252F4tiuNkPY6JupuKlPhZKM%252FQuantum-AI-PennyLane.png%3Falt%3Dmedia%26token%3D7389dbaa-285d-4a77-baea-bd0fa20e7bb3&width=300&dpr=3&quality=100&sign=7eeaba15&sv=2" width="500" height="707"/>

## 책 소개

머신러닝을 진지하게 다뤄본 사람이라면 누구나 한 번쯤 양자 컴퓨팅에 호기심을 가진 순간이 있다. 큐비트가 동시에 0과 1일 수 있다는 설명을 들었거나, 얽힘이 "공간을 초월한 상관관계"라는 비유를 본 적이 있을 것이다. 그러나 막상 IBM Quantum이나 Qiskit 튜토리얼을 열어보면 첫 페이지부터 Bra-ket 표기와 Hermitian 연산자가 쏟아져 나오고, 한 줄짜리 회로가 어떻게 신경망의 한 레이어로 변환되는지는 어디에도 적혀 있지 않다. "양자 머신러닝(QML)"이라는 키워드로 검색하면 수백 편의 논문이 나오지만, 정작 "이 회로를 학습시키려면 PyTorch의 어디에 끼워 넣어야 하나"라는 가장 실용적인 질문에 답하는 자료는 의외로 적다.

이것이 ML 엔지니어가 양자 컴퓨팅 앞에서 멈춰서는 가장 흔한 지점이다. 양자물리의 추상화와 머신러닝의 추상화 사이에 "미분 가능한 양자 회로"라는 다리가 있다는 사실을, 그리고 그 다리를 건너는 도구가 이미 성숙해 있다는 사실을 모르기 때문이다. 우리는 큐비트와 텐서 사이를 오가는 chain rule을 손으로 유도하다 지치고, 시뮬레이터와 하드웨어를 분리해 다루는 보일러플레이트에 묻혀 본질을 놓친다. 어느 시점에서 깨닫게 된다. 이것은 양자물리의 문제가 아니라 추상화의 문제다. 본 책은 그 깨달음에서 출발한다.
2018년 Xanadu가 공개한 PennyLane은 단순한 양자 시뮬레이터가 아니라, 양자 회로를 자동미분 가능한 함수로 다루기 위해 설계된 최초의 본격적 프레임워크다. Mitarai 등(2018)과 Schuld 등(2019)이 정립한 parameter-shift rule이 그 수학적 기반이고, 그 위에 PyTorch · JAX · TensorFlow · NumPy 같은 ML 인터페이스를 "한 줄로 갈아 끼울 수 있게" 만든 추상화가 그 공학적 핵심이다. 이로써 양자 회로는 마침내 신경망의 한 레이어처럼 다뤄질 수 있게 되었고, QML이라는 분야 전체가 "이론적 가능성"에서 "실험할 수 있는 코드"의 단계로 넘어갔다.

본 책은 양자 컴퓨팅을 처음 접하는 ML 경험자, PennyLane 튜토리얼을 가볍게 따라가본 개발자, 그리고 자신의 도메인에 QML을 도입해보려는 실무자를 동시에 대상으로 한다. 양자물리 전공자가 아니더라도, 머신러닝 경험이 어느 정도 있다면 본 책을 끝까지 따라갈 수 있도록 "직관 → 수식 → 코드"의 동일한 패턴을 22개 챕터에 일관되게 적용했다.

독자는 예제의 ansatz 구조·큐비트 수·학습률·인코딩 방식을 조정하거나 자신의 데이터로 실험해 보는 것을 적극 권장한다. Moons 데이터나 H₂ 분자를 수동적으로 따라가는 것과, 자신이 매일 다루는 분류 문제나 도메인 데이터에 직접 적용해보는 것은 전혀 다른 학습 경험이다.


## 목 차

저자 소개

Table of Contents (목차)

서문: 들어가며

Part 0: PennyLane을 위한 수학적 기초

Part 1: QML을 위한 PennyLane 기본
- Chapter 1. PennyLane 시작하기
- Chapter 2. 큐비트와 양자 게이트
- Chapter 3. QNode와 디바이스
- Chapter 4. 매개변수화 양자 회로 (PQC)
- Chapter 5. 자동미분과 최적화
- Chapter 6. 고전 데이터의 양자 인코딩
- Chapter 7. 양자 모델 = 푸리에 급수

Part 2: Quantum Machine Learning
- Chapter 8. 변분 양자 분류기 (VQC)
- Chapter 9. Quantum Kernel & SVM
- Chapter 10. Quantum Regression
- Chapter 11. QAOA — 조합 최적화
- Chapter 12. VQE — 양자 화학과 양자 데이터
- Chapter 13. Barren Plateau와 표현력
- Chapter 14. Quantum GAN
- Chapter 15. Geometric Quantum Machine Learning

Part 3: Quantum Deep Learning with PyTorch
- Chapter 16. PennyLane × PyTorch 연동
- Chapter 17. 하이브리드 아키텍처 설계
- Chapter 18. Quanvolutional Neural Networks
- Chapter 19. Quantum Transfer Learning
- Chapter 20. Quantum LSTM — 시계열 예측
- Chapter 21. 학습 팁과 성능 최적화
- Chapter 22. PannyLane 캡스톤 프로젝트


## E-Book 구매

- Yes24: https://www.yes24.com/product/goods/191823010
- 교보문고: https://ebook-product.kyobobook.co.kr/dig/epd/ebook/E000013141842
- 알라딘: https://aladin.kr/p/FStKe

## Github 코드: 

https://github.com/no-wave/quantum-ai-pennylane




