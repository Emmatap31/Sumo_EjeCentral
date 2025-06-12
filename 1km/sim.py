import traci
import sumolib

#Esta función modifica el programa de los semáforos
def new_logic(ggrr, yyrr, rrgg, rryy):

    tls_ids = traci.trafficlight.getIDList()
    id_semaforo = tls_ids[0]
    programa = traci.trafficlight.getAllProgramLogics(id_semaforo)
    
    from traci._trafficlight import Logic, Phase

    new_logic = Logic(programID="nuevo_programa", type=0,currentPhaseIndex=0,phases=[Phase(duration=ggrr, state="GGrr", minDur=ggrr, maxDur=ggrr ), Phase(duration=yyrr, state='yyrr', minDur=yyrr, maxDur=yyrr), Phase(duration=rrgg, state='rrGG', minDur=rrgg, maxDur=rrgg), Phase(duration=rryy, state='rryy', minDur=rryy, maxDur=rryy)])

    traci.trafficlight.setProgramLogic(id_semaforo, new_logic)

    traci.trafficlight.setProgram(id_semaforo, "nuevo_programa")

    #print(traci.trafficlight.getCompleteRedYellowGreenDefinition(id_semaforo))

import numpy as np
    
#Esta función define el número de autos que van a aparecer(en promedio) en la ruta r_0(n0) y en la ruta r_1(n1)
def n_vehiculos(n0, n1):

    t0=0
    t1=0
    for i in range(n0):

        t0+= np.random.exponential(scale=3600/n0)

        traci.vehicle.add(
            vehID=f"veh_r0_{i}",
            routeID="r_0", 
            depart=t0
        )

    for i in range(n1):

        t1+= np.random.exponential(scale=3600/n1)

        traci.vehicle.add(
            vehID=f"veh_r1_{i}",
            routeID="r_1",
            depart=t1
        )

#Aquí se llevan a cabo las simulaciones
import random as rd
import pandas as pd
import traci
import sumolib

avgtimes=[]#tiempos promedio
ggrrs = []#(verde Norte-Sur)
yyrrs = []#(amarillo Norte-Sur)
rrggs = []# (verde Este-Oeste)
rryys = []#(amarillo Este-Oeste)
n0s = []#Tiempos ruta 0
n1s = []#Tiempos ruta 1
num_simulations = 1000

#Flujos de autos que se van a utilizar
f0 = [i for i in range(1500,25000, 2000)]
f1 = [i for i in range(2000,30000, 2000)]


for i in range(num_simulations):

    n0 = rd.choice(f0)
    n1 = rd.choice(f1)

    n0s.append(n0)
    n1s.append(n1)

    sumoBinary = "sumo"
    sumoCmd = [sumoBinary, "-c", "prueba.sumocfg"]

    traci.start(sumoCmd)

    n_vehiculos(n0, n1)

    ggrr = rd.randint(30,70) #(verde Norte-Sur) 
    yyrr = rd.randint(3,7) #(amarillo Norte-Sur)
    rrgg = rd.randint(30,70) # (verde Este-Oeste)
    rryy = rd.randint(3,7) #(amarillo Este-Oeste)

    ggrrs.append(ggrr)
    yyrrs.append(yyrr)
    rrggs.append(rrgg)
    rryys.append(rryy)

    new_logic(ggrr, yyrr, rrgg, rryy)
    
    depart_times = {}
    arrival_times = {}
    simulation_durations = {}

    for step in range(3600):
        traci.simulationStep()

        departed_vehicles = traci.simulation.getDepartedIDList()
        current_time = traci.simulation.getTime()
        for veh_id in departed_vehicles:
            if veh_id not in depart_times:
                depart_times[veh_id] = current_time
        
        

        arrived_vehicles = traci.simulation.getArrivedIDList()
        current_time = traci.simulation.getTime()
        for veh_id in arrived_vehicles:
            if veh_id not in arrival_times:
                arrival_times[veh_id] = current_time



    for veh_id, arrival_time in arrival_times.items():
        if veh_id in depart_times:
            departure_time = depart_times[veh_id]
            duration = arrival_time - departure_time
            simulation_durations[veh_id] = duration

    average = sum(simulation_durations.values())/len(simulation_durations)

    avgtimes.append(average)

    traci.close()
    print(i+1)
data = {
    'Simulation' : range(1, num_simulations + 1),
    'Tiempo promedio' : avgtimes,
    'GGrr (s)' : ggrrs,
    'yyrr (s)' : yyrrs,
    'rrGG (s)' : rrggs,
    'rryy (s)' : rryys,
    'Autos r_0': n0s,
    'Autos r_1': n1s
}

df100 = pd.DataFrame(data)
df100



#Regresiones--------------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

data = pd.read_csv('Valores.csv')

variables_independientes = ['GGrr (s)','yyrr (s)','rrGG (s)','rryy (s)','Autos r_0', 'Autos r_1']
variable_dependiente = 'Tiempo promedio'

X = data[variables_independientes]
y = data[variable_dependiente]

#Se dividen los datos en conjunto de entrenamiento y de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Se entrena el modelo de regresión polinomial de grado 3
modelo3 = make_pipeline( PolynomialFeatures(degree=3), LinearRegression())
modelo3.fit(X_train, y_train)
y_pred3= modelo3.predict(X_test)

#Se evalúa el modelo
mse = mean_squared_error(y_test, y_pred3)
r2 = r2_score(y_test, y_pred3)
print(f"Error cuadrático medio: {mse}")
print(f"R^2 = {r2}")


#Optimización---------------------------------------------------------------
import numpy as np
from scipy.optimize import minimize



def modelo_predict(vars_to_optimize, r0, r1):
    GGrr, yyrr, rrGG, rryy = vars_to_optimize
    df_input = pd.DataFrame([{
        'GGrr (s)': GGrr,
        'yyrr (s)': yyrr,
        'rrGG (s)': rrGG,
        'rryy (s)': rryy,
        'Autos r_0': r0,
        'Autos r_1': r1
    }])
    return modelo3.predict(df_input)[0]

# Esta función hace la optimización
def minimizar_modelo(r0, r1):
    def objective(x):  # x contiene GGrr, yyrr, rrGG, rryy
        return modelo_predict(x, r0, r1)

    x0 = np.array([35, 3.5, 35, 3.5])  # valores iniciales arbitrarios
    bounds = [(10, 60), (2, 5), (10, 60), (2, 5)]  

    result = minimize(objective, x0, bounds=bounds)

    if result.success:
        GGrr, yyrr, rrGG, rryy = result.x
        print(f"Valores óptimos para r0 = {r0}, r1 = {r1}:")
        print(f"GGrr  = {GGrr}")
        print(f"yyrr  = {yyrr}")
        print(f"rrGG  = {rrGG}")
        print(f"rryy  = {rryy}")
        print(f"Valor mínimo del modelo: {result.fun}")
        return result.x, result.fun
    else:
        raise RuntimeError("Optimización fallida:", result.message)