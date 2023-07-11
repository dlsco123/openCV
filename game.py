import pygame
import os
import sys
import random
import cv2
import numpy as np
from keras.models import load_model

# Pygame 초기화
pygame.init()

# 화면 크기 설정
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))

# 플레이어 설정
player_size = 50
player_pos = [screen_width / 2, screen_height - player_size - 20]

# 총알 설정
bullet_size = 50
bullet_pos = [random.randrange(0, screen_width - bullet_size), 0]
bullet_list = [bullet_pos]

# 게임 속도
speed = 1

# Teachable Machine 모델 불러오기
model_path = "D:/models/game/keras_model.h5"
assert os.path.exists(model_path), f"Model file not found at {model_path}"
model = load_model(model_path)


# 웹캠 설정
cap = cv2.VideoCapture(0)

# 움직임 속도
move_speed = 5

# 게임 루프
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill((0, 0, 0))

    # 웹캠에서 이미지를 얻고, 모델에 입력으로 전달
    ret, frame = cap.read()
    if not ret:
        continue
    # 이미지 전처리 (여기서는 간단하게 크기만 조정했지만, 실제로는 더 복잡한 전처리가 필요할 수 있습니다)
    frame = cv2.resize(frame, (224, 224))  # 모델에 맞게 이미지 크기 조정
    frame = np.expand_dims(frame, axis=0)

    # 모델에 이미지를 입력하여 예측값을 얻음
    pred = model.predict(frame)
    class_idx = np.argmax(pred)

    # 분류된 클래스에 따라 플레이어 이동
    if class_idx == 0:  # 'left'
        player_pos[0] -= move_speed
    elif class_idx == 1:  # 'right'
        player_pos[0] += move_speed

    
    # 플레이어의 위치가 화면 밖으로 나가지 않도록 제한
    player_pos[0] = max(0, min(screen_width - player_size, player_pos[0]))

    # 플레이어 그리기
    pygame.draw.rect(screen, (0, 255, 0), (player_pos[0], player_pos[1], player_size, player_size))

    # 총알 속도 업데이트
    if bullet_pos[1] >= 0 and bullet_pos[1] < screen_height:
        bullet_pos[1] += speed
    else:
        bullet_pos[0] = random.randrange(0, screen_width - bullet_size)
        bullet_pos[1] = 0

    # 총알 그리기
    pygame.draw.rect(screen, (255, 0, 0), (bullet_pos[0], bullet_pos[1], bullet_size, bullet_size))

    # 총알이 플레이어를 맞추면 게임 오버
    if bullet_pos[1] >= player_pos[1] and bullet_pos[0] in range(int(player_pos[0]), int(player_pos[0] + player_size)):
        pygame.quit()
        sys.exit()

    pygame.display.update()

cap.release()
cv2.destroyAllWindows()