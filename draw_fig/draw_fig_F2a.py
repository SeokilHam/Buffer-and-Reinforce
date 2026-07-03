import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as font_manager

# 1. 데이터 정의 (이미지 기반 추정값)
x = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
y_llama3 = [0.64, 0.79, 0.95, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
y_gemma = [0.58, 0.60, 0.92, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
y_llama2 = [0.82, 0.87, 0.95, 0.98, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]

# 2. 그래프 스타일 설정 (가독성 향상)
plt.figure(figsize=(8, 6)) # 그림 크기를 키움

# 폰트 설정
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# 3. 플롯 그리기 (선과 점을 더 굵고 크게)
plt.plot(x, y_llama3, label='Llama3-8B-Instruct', marker='o', markersize=9, color='#1f77b4', linewidth=3)
plt.plot(x, y_gemma, label='Gemma-2B-it', marker='o', markersize=9, color='#ff7f0e', linewidth=3)
plt.plot(x, y_llama2, label='Llama2-7B-chat', marker='o', markersize=9, color='#2ca02c', linewidth=3)

# 4. 축 레이블 및 틱 설정 (폰트 크기 확대)
plt.xlabel(r'$\lambda$', fontsize=30, fontweight='bold')
plt.ylabel('Refusal Rate', fontsize=30, fontweight='bold')

plt.xticks(np.arange(0, 1.85, 0.25), fontsize=22, fontweight='bold')
plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0], fontsize=22, fontweight='bold')


# 축 범위
plt.ylim(0.56, 1.02)
plt.xlim(-0.1, 1.85)

# 5. 범례 설정
font_prop = font_manager.FontProperties(family='serif', size=22, weight='bold')
plt.legend(prop=font_prop, loc='center right', frameon=True, edgecolor='#d3d3d3', framealpha=1)

# 6. 격자 추가 (값 읽기 편하게)
plt.grid(True, which='major', linestyle='--', alpha=0.3)

# 7. 레이아웃 조정 및 저장
plt.tight_layout()
plt.savefig('acc.pdf', dpi=300, bbox_inches='tight', pad_inches=0) # 고해상도 PNG 저장
plt.show()