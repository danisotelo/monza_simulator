import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


# Universo de discurso para el error
universe_e = np.linspace(-0.2, 0.2, 100) 

# Funciones de pertenencia para el error
ENG = fuzz.trapmf(universe_e, [-0.2, -0.2, -0.06, -0.04])  
ENP = fuzz.trimf(universe_e, [-0.06, -0.03, 0])  
EC = fuzz.trimf(universe_e, [-0.02, 0, 0.02])  
EPP = fuzz.trimf(universe_e, [0, 0.03, 0.06])  
EPG = fuzz.trapmf(universe_e, [0.04, 0.06, 0.2, 0.2])  

# Universo de discurso para el ángulo
universe_s = np.linspace(-0.25, 0.25, 100) 

# Funciones de pertenencia para el ángulo
ANG = fuzz.trapmf(universe_s, [-0.25, -0.25, -0.1, -0.07])  
ANP = fuzz.trimf(universe_s, [-0.1, -0.05, 0])  
AC = fuzz.trimf(universe_s, [-0.03, 0, 0.03])  
APP = fuzz.trimf(universe_s, [0, 0.05, 0.1])  
APG = fuzz.trapmf(universe_s, [0.07, 0.1, 0.25, 0.25])  

# Visualización
plt.figure(figsize=(10, 5))
plt.plot(universe_e, ENG, label="Error Negativo Grande (ENG)")
plt.plot(universe_e, ENP, label="Error Negativo Pequeño (ENP)")
plt.plot(universe_e, EC, label="Error Cero (EC)")
plt.plot(universe_e, EPP, label="Error Positivo Pequeño (EPP)")
plt.plot(universe_e, EPG, label="Error Positivo Grande (EPG)")
plt.xlabel("Error")
plt.ylabel("Grado de Pertenencia")
plt.title("Conjuntos Difusos para Error")
plt.legend()
plt.grid(True)
plt.show()

e0 = 0.9  # el valor del ratio de cambio de error actual

# Fuzzificar: encontrar la pertenencia de e0 a cada conjunto difuso de entrada
val_ENG = fuzz.interp_membership(universe_e, ENG, e0)
val_ENP = fuzz.interp_membership(universe_e, ENP, e0)
val_EC = fuzz.interp_membership(universe_e, EC, e0)
val_EPP = fuzz.interp_membership(universe_e, EPP, e0)
val_EPG = fuzz.interp_membership(universe_e, EPG, e0)

# Visualicemos los valores
print('val_ENG =',val_ENG)
print('val_ENP =',val_ENP)
print('val_EC =',val_EC)
print('val_EPP =',val_EPP)
print('val_EPG =',val_EPG)

# Gráficas para visualizar sus posiciones
plt.figure(figsize=(10, 5))
plt.plot(universe_e, ENG, label="ENG")
plt.plot(universe_e, ENP, label="ENP")
plt.plot(universe_e, EC, label="EC")
plt.plot(universe_e, EPP, label="EPP")
plt.plot(universe_e, EPG, label="EPG")
plt.legend(loc='best')

plt.plot([e0, e0], [0.0, 1.0], linestyle="--")
plt.plot(e0, val_ENG, 'x')
plt.plot(e0, val_ENP, 'x')
plt.plot(e0, val_EC, 'x')
plt.plot(e0, val_EPP, 'x')
plt.plot(e0, val_EPG, 'x')
plt.xlabel('error')
plt.ylabel('$\mu (e)$')
plt.show()


# Calcular las funciones cortadas
ANGp = np.fmin(val_ENG, ANG)
ANPp = np.fmin(val_ENP, ANP)
ACp = np.fmin(val_EC, AC)
APPp = np.fmin(val_EPP, APP)
APGp = np.fmin(val_EPG, APG)

# Unificar corte
Ap = np.fmax(np.fmax(np.fmax(ANGp, ANPp), np.fmax(ACp, APPp)), APGp)

# y graficamos
plt.figure(figsize=(10, 5))
plt.plot(universe_s, ANG, label="ANG")
plt.plot(universe_s, ANP, label="ANP")
plt.plot(universe_s, AC, label="AC")
plt.plot(universe_s, APP, label="APP")
plt.plot(universe_s, APG, label="APG")
plt.plot(universe_s, Ap, label="Ap", linewidth=3)
plt.legend(loc='best')
plt.show()

# Aplicamos defuzzificación
out_centroid = fuzz.defuzz(universe_s, Ap, 'centroid')
out_bisector = fuzz.defuzz(universe_s, Ap, 'bisector')
out_MOM = fuzz.defuzz(universe_s, Ap, 'mom')
out_SOM = fuzz.defuzz(universe_s, Ap, 'som')
out_LOM = fuzz.defuzz(universe_s, Ap, 'lom')

# Gráficos para comparar
plt.figure(figsize=(10, 5))
plt.plot(universe_s, Ap, linewidth=2.5, linestyle="-", label="Ap")
plt.plot([out_centroid, out_centroid], [0, 1], linestyle=":", label="out_centroid")
plt.plot([out_bisector, out_bisector], [0, 1], linestyle=":", label="out_bisector")
plt.plot([out_MOM, out_MOM], [0, 1], linestyle=":", label="out_MOM")
plt.plot([out_SOM, out_SOM], [0, 1], linestyle=":", label="out_SOM")
plt.plot([out_LOM, out_LOM], [0, 1], linestyle=":", label="out_LOM")
plt.legend(loc='best')
plt.show()
