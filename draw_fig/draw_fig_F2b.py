import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as font_manager

# 1. 데이터 정의 (이미지 기반 추정값)
x = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]

# Llama3-8B-Instrcut (Blue)
y_llama3 = [0.84, 0.74, 0.52, 0.27, 0.15, 0.05, 0.03, 0.06, 0.12, 0.12]

# Gemma-2B-it (Orange)
y_gemma = [0.89, 0.83, 0.64, 0.50, 0.49, 0.43, 0.32, 0.21, 0.18, 0.26]

# Llama2-7B-chat (Green)
y_llama2 = [0.88, 0.78, 0.68, 0.60, 0.54, 0.50, 0.48, 0.39, 0.22, 0.18]

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
plt.ylabel('Diversity', fontsize=30, fontweight='bold')

plt.xticks(np.arange(0, 1.85, 0.25), fontsize=22, fontweight='bold')
plt.yticks(np.arange(0.0, 1.0, 0.2), fontsize=22, fontweight='bold')

# 축 범위 설정
plt.ylim(-0.01, 0.92)
plt.xlim(-0.1, 1.85)

# 5. 범례 설정
font_prop = font_manager.FontProperties(family='serif', size=22, weight='bold')
plt.legend(prop=font_prop, loc='upper right', frameon=True, edgecolor='#d3d3d3', framealpha=1)

# 6. 격자 추가
plt.grid(True, which='major', linestyle='--', alpha=0.3)

# 7. 저장 및 출력
plt.tight_layout()
plt.savefig('diverse.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()