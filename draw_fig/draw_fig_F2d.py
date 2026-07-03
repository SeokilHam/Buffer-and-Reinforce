import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as font_manager

# 1. 데이터 정의 (이미지 기반 추정값)
x = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]


# 샘플 데이터 (원하는 대로 값 수정 가능)
# x = np.linspace(-1.0, 0.0)                   # fixed answer rate
x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0])
y_llama3 = np.array([0.37, 0.57, 0.65, 0.64, 0.67, 0.65]) # 파란선  # Llama3
y_gemma = np.array([0.05, 0.09, 0.10, 0.08, 0.12, 0.11])  # 주황선  # Gemma
y_llama2 = np.array([0.07, 0.07, 0.09, 0.12, 0.10, 0.11])  # 초록선  # Llama2

# 2. 그래프 스타일 설정
plt.figure(figsize=(8, 6)) # 그림 크기를 키움

# 폰트 설정
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# 3. 플롯 그리기
plt.plot(x, y_llama3, label='Llama3-8B-Instruct', marker='o', markersize=9, color='#1f77b4', linewidth=3)
plt.plot(x, y_gemma, label='Gemma-2B-it', marker='o', markersize=9, color='#ff7f0e', linewidth=3)
plt.plot(x, y_llama2, label='Llama2-7B-chat', marker='o', markersize=9, color='#2ca02c', linewidth=3)

# 4. 축 레이블 및 틱 설정
plt.xlabel(r'$\lambda$', fontsize=30, fontweight='bold')
plt.ylabel('Accuracy', fontsize=30, fontweight='bold')

plt.ylim(0, 1.0)
plt.xticks(fontsize=22, fontweight='bold')
plt.yticks(fontsize=22, fontweight='bold')


# 5. 범례 설정
font_prop = font_manager.FontProperties(family='serif', size=20, weight='bold')
plt.legend(prop=font_prop, loc='upper left', frameon=True, edgecolor='#d3d3d3', framealpha=1)

# 6. 격자 추가
plt.grid(True, which='major', linestyle='--', alpha=0.3)

# 7. 저장 및 출력
plt.tight_layout()
plt.savefig('refusal_gsm8k.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()