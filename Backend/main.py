import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import random
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# === STEP 1: User Input and ML model  ===
# Load data
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv('pima-indians-diabetes.data.csv', names=columns)

X = df[['Glucose', 'BMI']]   # Features
y = df['Outcome']            # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

Glucose = float(input("Enter Glucose level: "))
BMI = float(input("Enter BMI: "))

# === STEP 2: Risk Function ===
def risk_function(x, y):
    # If scalar input, handle directly
    if np.isscalar(x) and np.isscalar(y):
        df_input = pd.DataFrame({'Glucose': [x], 'BMI': [y]})
        prob = model.predict_proba(df_input)[0][1]  # Probability of being diabetic
        return prob

    # Else, handle vector input (e.g., for meshgrid or animation)
    x_flat = x.ravel()
    y_flat = y.ravel()
    df_input = pd.DataFrame({'Glucose': x_flat, 'BMI': y_flat})
    probs = model.predict_proba(df_input)[:, 1]
    return probs.reshape(x.shape)

# === STEP 3: Cuckoo Search Algorithm Setup ===
class Cuckoo:
    def __init__(self, position):
        self.position = position
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        x, y = self.position
        return risk_function(x, y)

class CuckooSearch:
    def __init__(self, num_cuckoos=20, max_iterations=100, beta=1.5):
        self.num_cuckoos = num_cuckoos
        self.max_iterations = max_iterations
        self.beta = beta
        self.cuckoos = [Cuckoo(self.random_position()) for _ in range(num_cuckoos)]
        self.history = []

    def random_position(self):
        return [
            random.uniform(50, 200),  # Glucose range
            random.uniform(10, 50)   # BMI range
        ]

    def levy_flight(self):
        sigma = (math.gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2) /
                 (math.gamma((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2))) ** (1 / self.beta)
        u = np.random.normal(0, sigma, 2)
        v = np.random.normal(0, 1, 2)
        step = u / (np.abs(v) ** (1 / self.beta))
        return step

    def optimize(self):
        for _ in range(self.max_iterations):
            for i in range(self.num_cuckoos):
                step = self.levy_flight()
                current_pos = np.array(self.cuckoos[i].position)
                new_pos = current_pos + step
                # Keep within bounds
                new_pos[0] = np.clip(new_pos[0], 50, 200)
                new_pos[1] = np.clip(new_pos[1], 10, 50)
                new_cuckoo = Cuckoo(new_pos.tolist())
                if new_cuckoo.fitness < self.cuckoos[i].fitness:
                    self.cuckoos[i] = new_cuckoo
            self.cuckoos.sort(key=lambda c: c.fitness)
            best = self.cuckoos[0]
            self.history.append(best.position + [best.fitness])
        return self.history

# === STEP 4: Run CSA ===
csa = CuckooSearch(num_cuckoos=25, max_iterations=60)
search_history = csa.optimize()

# === STEP 5: Create Grid for Plot ===
x = np.linspace(50, 200, 100)
y = np.linspace(10, 50, 100)
X, Y = np.meshgrid(x, y)
Z = risk_function(X, Y)

# === STEP 6: 2D Animation ===
fig2d, ax2d = plt.subplots()
contour = ax2d.contourf(X, Y, Z, cmap='viridis')
point2d, = ax2d.plot([], [], 'ro', markersize=6)
ax2d.set_title("CSA Optimization in 2D")
ax2d.set_xlabel("Glucose")
ax2d.set_ylabel("BMI")

def init_2d():
    point2d.set_data([], [])
    return point2d,

def animate_2d(i):
    gx, bm, _ = search_history[i]
    point2d.set_data([gx], [bm])
    return point2d,

ani2d = animation.FuncAnimation(fig2d, animate_2d, init_func=init_2d,
                                frames=len(search_history), interval=150, blit=True)

# === STEP 7: 3D Animation ===
fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.7)
point3d, = ax3d.plot([Glucose], [BMI], [risk_function(Glucose, BMI)], 'go', label='Start', markersize=6)
current_point, = ax3d.plot([], [], [], 'ro', markersize=6)
ax3d.set_xlabel("Glucose")
ax3d.set_ylabel("BMI")
ax3d.set_zlabel("Risk")
ax3d.set_title("CSA Optimization in 3D")

def animate_3d(i):
    gx, bm, risk = search_history[i]
    current_point.set_data([gx], [bm])
    current_point.set_3d_properties([risk])
    ax3d.view_init(elev=30, azim=i*3)
    return current_point,

ani3d = animation.FuncAnimation(fig3d, animate_3d, frames=len(search_history),
                                interval=150, blit=False)

# === STEP 8: Show Animations ===
plt.show()
