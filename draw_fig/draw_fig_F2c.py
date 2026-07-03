import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as font_manager

# 1. 데이터 정의 (이미지 기반 추정값)
x = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
y_llama3 = [0.64, 0.79, 0.95, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
y_gemma = [0.58, 0.60, 0.92, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
y_llama2 = [0.82, 0.87, 0.95, 0.98, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]

# 샘플 데이터 (원하는 대로 값 수정 가능)
x = np.linspace(-1.0, 0.0, 10)                   # fixed answer rate
x = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0])
y_llama3 = np.array([0.004, 0.008, 0.008, 0.028, 0.04, 0.112]) # 파란선  # Llama3
y_gemma = np.array([0.192, 0.272, 0.364, 0.400, 0.432, 0.52])  # 주황선  # Gemma
y_llama2 = np.array([0.004, 0.008, 0.016, 0.084, 0.276, 0.592])  # 초록선  # Llama2

# y_llama3_norm = (y_llama3-np.min(y_llama3))/(np.max(y_llama3)-np.min(y_llama3))
# y_gemma_norm = (y_gemma-np.min(y_gemma))/(np.max(y_gemma)-np.min(y_gemma))
# y_llama2_norm = (y_llama2-np.min(y_llama2))/(np.max(y_llama2)-np.min(y_llama2))

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

plt.ylim(0, 1.0)
plt.xticks(fontsize=22, fontweight='bold')
plt.yticks(fontsize=22, fontweight='bold')


# 5. 범례 설정
font_prop = font_manager.FontProperties(family='serif', size=22, weight='bold')
plt.legend(prop=font_prop, loc='upper left', frameon=True, edgecolor='#d3d3d3', framealpha=1)

# 6. 격자 추가 (값 읽기 편하게)
plt.grid(True, which='major', linestyle='--', alpha=0.3)

# 7. 레이아웃 조정 및 저장
plt.tight_layout()
plt.savefig('refusal_xstest.pdf', dpi=300, bbox_inches='tight', pad_inches=0) # 고해상도 PNG 저장
plt.show()