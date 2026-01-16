from data.utils import load_data
from physics import TrainPhysics
from environment import TrainEnv
from cm_analyzer import CMAnalyzer

def main():
    # 1. Load Data
    grades, limits, curves = load_data()
    
    # 2. Initialize Physics
    physics = TrainPhysics()
    
    # 3. Initialize Environment
    env = TrainEnv(physics, grades, limits, curves)
    
    # 4. Run CM Analysis
    analyzer = CMAnalyzer(env)
    analyzer.run(episodes=25000)
    
    print("\nProcess Complete.")
    print("Please check the 'results_cm' folder for the plot.")

if __name__ == "__main__":
    main()