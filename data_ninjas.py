import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMRegressor
import lightgbm as lgb
from copy import deepcopy
from scipy.stats import median_test
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold,ShuffleSplit,GridSearchCV

"""
Funcion utilizada para hacer nuevos grupos para SocioDemo1 en funcion de la mediana de los poderes adquisitivos
(elementos en un mismo grupo tienen una mediada similar).Guarda un diccionario en formato pickle mapeando 
cada valor de SocioDemo1 al indice del grupo asignado.
Parametros:
limit:  p-valor a partir del cual se supone una igualdad de medianas
"""


def generate_cluster_dicts(limit=0.99):
	df = pd.read_csv('Dataset_Salesforce_Predictive_Modelling_TRAIN.txt', sep=",",header=0,engine="c",
		dtype= {"Socio_Demo_01":"category"})

	df["Socio_Demo_01"] = df["Socio_Demo_01"].replace(np.nan, 'undefined', regex=True)
	groups_dict = make_median_clust(df,"Socio_Demo_01",p_lim=limit)
	pickle.dump( groups_dict, open( "cluster_dict.p", "wb" ) )



"""
Funcion que contiene la logica del algoritmo que realiza clustering en funcion de las medianas usando el
test de Mood.
Parametros:
df: dataframe de pandas
column2clust: columna con la que se quiere hacer clustering (en nuestro caso SocioDemo1)
p_lim: p-valor para no rechazar que dos poblaciones tienen la misma mediana
"""
def make_median_clust(df,column2clust,p_lim=0.99):
	values = list(set(df[column2clust]))
	print("Oringinal size: ", len(values))
	ori_dict = dict()

	for value in values:
		conditioned_values = df.loc[df[column2clust] == value,"Poder_Adquisitivo"].values
		ori_dict[value] = conditioned_values

	final_group = defaultdict(int)
	n_grupo = 1
	for value in values:
		if final_group[value] == 0:
			final_group[value] = n_grupo
			for value2 in values:
				if value != value2:
					return_tuple = median_test(ori_dict[value],ori_dict[value2])
					if return_tuple[1] >= p_lim:
						if final_group[value2] != 0:
							print("Ya estaba en un grupo! ", value2)
						else:
							final_group[value2] = n_grupo
			n_grupo +=1
	print("Diccionario: ", final_group)
	print("Num grupos:", len(set(final_group.values())))
	return final_group


"""
Diferentes ejecuciones pueden generar diferentes clusters de SocioDemo1 , por ello
hemos decidido fijar un diccionario cuyas variables son los valores de SocioDemo1
y los valores son los indices de los clusters encontrados. 
Este diccionario se ha obtenido con un p-valor de 0.99
"""

clust_dictionatry = {'01111': 1, '03629': 1, '05822': 1, '02621': 1, '03531': 1, '04444': 1, '02464': 1, '02453': 1, '08420': 1, '07532': 1, '03324': 1, '01322': 1, '01429': 1, '09490': 2, '02157': 2, '02625': 2, '08154': 2, '01325': 2, '02427': 2, '09443': 2, '03539': 2, '05824': 2, '08153': 2, '03327': 2, '02825': 2, '02426': 2, '04446': 2, '08155': 2, '03716': 2, '07706': 2, '08113': 2, '07892': 2, '06204': 2, '03714': 2, '07405': 2, '07131': 3, '02324': 3, '07311': 3, '07291': 3, '08156': 3, '02482': 3, '03323': 3, '03715': 3, '07837': 4, '03123': 5, '02911': 5, '03160': 5, '02630': 5, '03403': 5, '02414': 5, '03133': 5, '03712': 5, '02437': 5, '02466': 5, '03312': 5, '03535': 5, '07223': 5, '02936': 5, '02151': 5, '02592': 5, '03314': 5, '02652': 5, '02454': 5, '03814': 5, '05993': 6, '03203': 6, '08141': 6, '02433': 6, '03201': 6, '02932': 6, '07111': 7, '08144': 7, '07112': 7, '05420': 7, '07617': 7, '03339': 7, '05491': 7, '05300': 7, '07192': 8, '08143': 8, '08331': 8, '07615': 8, '03721': 8, '08202': 8, '03202': 8, '07324': 8, '04500': 8, '04442': 8, '03142': 8, '03831': 8, '03207': 8, '08340': 8, '07232': 8, '05622': 8, '08122': 8, '04309': 8, '04111': 8, '07122': 8, '03739': 8, '03612': 8, '02721': 8, '09603': 8, '08322': 8, '06201': 8, '07510': 8, '08199': 8, '05210': 8, '03521': 8, '08411': 8, '00020': 8, '04112': 8, '04301': 8, '07315': 8, '07250': 8, '03143': 8, '03833': 8, '08332': 8, '05932': 9, '03813': 9, '02325': 9, '03134': 9, '02611': 9, '02713': 9, '05892': 9, '05991': 10, '02123': 10, '02822': 10, '02469': 10, '01326': 10, '02465': 10, '02719': 10, '02442': 10, '02613': 10, '01211': 11, '03121': 11, '05921': 11, '03401': 11, '02451': 11, '02921': 11, '02431': 11, '02461': 11, '02472': 11, '02937': 12, '09512': 12, '09995': 12, '02939': 12, '07616': 12, '05220': 12, '09410': 12, '09210': 12, '05840': 12, '04223': 12, '03325': 12, '09229': 12, '07121': 12, '02933': 12, '01419': 12, '09993': 12, '05811': 12, '04412': 12, '09820': 12, '07612': 12, '07834': 12, '02935': 12, '02153': 12, '03331': 12, '01422': 12, '05721': 12, '07708': 12, '07835': 12, '07611': 12, '09320': 12, '05893': 12, '05110': 12, '07832': 12, '07833': 12, '07812': 12, '09530': 12, '09432': 12, '05120': 12, 'undefined': 12, '05894': 12, '03722': 12, '06430': 12, '07705': 12, '07231': 12, '07702': 12, '07703': 12, '05823': 12, '04430': 12, '09223': 12, '05831': 12, '03724': 12, '05895': 13, '06202': 13, '08440': 13, '07211': 13, '02830': 13, '02326': 13, '02322': 13, '05899': 13, '07623': 13, '07811': 14, '02931': 14, '09812': 14, '07613': 15, '03632': 16, '02423': 16, '02412': 16, '01120': 17, '07212': 18, '09602': 18, '02154': 19, '08121': 19, '09441': 19, '08152': 19, '01411': 19, '04121': 19, '07891': 19, '06410': 19, '08192': 19, '07312': 19, '03731': 19, '04221': 19, '07622': 19, '07314': 19, '03812': 19, '05492': 19, '08151': 19, '02122': 20, '02483': 20, '01327': 20, '02462': 20, '03311': 20, '02416': 20, '02436': 20, '03152': 20, '03329': 21, '04123': 21, '03135': 21, '07707': 21, '03522': 21, '02622': 21, '04122': 21, '08142': 21, '02923': 21, '08111': 21, '09434': 21, '02653': 22, '07893': 22, '07221': 22, '03534': 22, '06421': 22, '05833': 22, '05941': 22, '08132': 22, '09431': 22, '03732': 22, '05999': 22, '03533': 22, '04222': 22, '06203': 22, '04422': 22, '07701': 22, '03614': 22, '08159': 22, '03523': 22, '08160': 22, '07899': 22, '03832': 22, '09811': 23, '09700': 23, '07323': 24, '03132': 25, '03155': 25, '02422': 25, '01221': 25, '02411': 25, '02912': 25, '08145': 25, '02312': 25, '03733': 25, '02329': 26, '03209': 27, '07531': 27, '03141': 27, '02824': 27, '01316': 28, '04445': 28, '04443': 28, '07193': 28, '05612': 28, '03125': 28, '07292': 28, '08133': 28, '03734': 28, '02156': 28, '03532': 29, '02252': 29, '07321': 29, '07619': 29, '02712': 30, '02922': 30, '02934': 31, '01323': 31, '02112': 31, '02111': 31, '01312': 31, '02452': 32, '03315': 32, '05923': 32, '08311': 32, '01314': 32, '03405': 32, '02711': 33, '02413': 33, '01313': 33, '03623': 33, '01315': 34, '03129': 34, '07404': 34, '09442': 35, '06423': 36, '02323': 36, '07191': 36, '07240': 36, '05629': 36, '05500': 36, '04423': 36, '05000': 37, '05493': 37, '07295': 37, '03321': 37, '05411': 37, '03153': 38, '03631': 38, '09511': 39, '05992': 39, '07894': 39, '03723': 39, '09222': 39, '04421': 40, '03205': 41, '08114': 41, '01311': 41, '03326': 41, '07533': 41, '02158': 41, '02624': 41, '02321': 41, '00012': 41, '08131': 41, '08191': 41, '02513': 42, '02159': 43, '03124': 44, '01509': 44, '02441': 44, '07521': 44, '03128': 45, '03811': 45, '03151': 45, '03110': 45, '03122': 46, '02612': 46, '03713': 46, '01324': 46, '01222': 47, '02722': 47, '09433': 48, '06422': 48, '08170': 48, '09541': 48, '08412': 49, '02484': 49, '03206': 49, '06205': 49, '07132': 49, '07618': 49, '03402': 50, '07831': 51, '09542': 51, '05412': 51, '05891': 51, '08112': 52, '08432': 52, '07322': 52, '08431': 52, '02435': 53, '02473': 53, '02240': 53, '02640': 54, '03126': 55, '05499': 56, '09520': 57, '03510': 58, '02155': 58, '08209': 59, '06209': 60, '07836': 60, '05621': 60, '07820': 61, '08193': 61, '07522': 62, '03131': 62, '01432': 63, '05922': 64, '01431': 65, '02421': 65, '01223': 65, '04411': 66, '05942': 66, '03316': 66, '08321': 66, '07294': 66, '02599': 67, '02220': 68, '02723': 68, '05910': 68, '02415': 69, '02432': 69, '02591': 69, '08312': 69, '02512': 69, '02434': 69, '02439': 69, '03204': 70, '02729': 70, '03127': 70, '09100': 71, '05832': 72, '05825': 72, '07401': 73, '04210': 73, '09601': 74, '03139': 75, '02130': 75, '07199': 76, '04424': 77, '03622': 78, '02463': 79, '07402': 79, '00011': 79, '01112': 79, '02810': 79, '02311': 80, '09992': 81, '05722': 81, '06120': 82, '09994': 83, '01329': 84, '02481': 84, '05812': 85, '03613': 86, '07313': 87, '09543': 87, '07621': 87, '02425': 88, '01212': 89, '02821': 89, '03621': 89, '02424': 89, '09221': 90, '03154': 91, '05710': 92, '07222': 93, '08333': 93, '03313': 94, '04113': 94, '03711': 94, '03404': 94, '01219': 95, '09310': 96, '09420': 96, '02651': 97, '02140': 98, '01113': 98, '01321': 98, '03317': 98, '02623': 99, '05821': 100, '08201': 100, '03820': 101, '01421': 102, '02152': 102, '02443': 103, '02121': 103, '06300': 104, '07709': 104, '02251': 105, '02823': 106, '07704': 107, '05611': 108, '06110': 109, '09991': 110, '03611': 111, '03322': 111, '0X301': 112, '04441': 113, '07293': 114, '05430': 114, '05931': 115, '07403': 116, '01501': 116, '02511': 117, '07614': 117, '02210': 118, '02230': 119, '02471': 120}



"""
Funcion utilizada para cargar y seleccionar datos de los datasets proporcionados
Parametros:
path: ruta en la que se encuentra el dataset segun el formato proporcionado para el concurso
is_test: distengue si el dataset se usa como entrenamiento o test
"""
def use_fram(path,is_test):
	df = pd.read_csv(path, sep=",",header=0,engine="c",dtype= {"Socio_Demo_01":"category"})

	df["Socio_Demo_01"] = df["Socio_Demo_01"].replace(np.nan, 'undefined', regex=True)


	#Obtenemos el ID del usuario para poder devolverlo si se trata de un dataset para test
	client_index = df["ID_Customer"]

	df = df.iloc[:,1:]


	#Uso de logaritmos
	#Evitamos valores negativos
	minimo_impcons1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	minimo_impcons2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	minimo_impsal1 = [0.0, 0.0, 0.0, 0.28, 2.36, 0.0, 0.0, 0.0, 0.0]
	minimo_impsal2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 45290.8, 0.0, 0.0]

	for i in range(1,10):
		minimo = minimo_impcons1.pop(0)
		df["Imp_Cons_0"+str(i)] = np.log(df["Imp_Cons_0"+str(i)] + minimo + 1)

	for i in range(11,18):
		minimo = minimo_impcons2.pop(0)
		df["Imp_Cons_"+str(i)] = np.log(df["Imp_Cons_"+str(i)] + minimo + 1)


	for i in range(1,10):
		minimo = minimo_impsal1.pop(0)
		df["Imp_Sal_0"+str(i)] = np.log(df["Imp_Sal_0"+str(i)] + minimo + 1)

	for i in range(10,22):
		minimo = minimo_impsal2.pop(0)
		df["Imp_Sal_"+str(i)] = np.log(df["Imp_Sal_"+str(i)] + minimo + 1)


	#Transformamos a tipo booleano si un individuo posee un producto o no 0 y 2 es false y 1 es true
	for i in range(2,10):
		df["Ind_Prod_0"+str(i)] = df["Ind_Prod_0"+str(i)].apply(lambda x: int(x==1))

	for i in range(10,25):
		df["Ind_Prod_"+str(i)] = df["Ind_Prod_"+str(i)].apply(lambda x: int(x==1))

	#Sumamos todas las variables Ind_Prod para obtener el numero total de productos de un cliente, se genera una
	#nueva variable (ver informe)
		
	df['Ind_Prod_01'] = df.apply(lambda row: row.Ind_Prod_01 + row.Ind_Prod_02+  row.Ind_Prod_03 + row.Ind_Prod_04 +  row.Ind_Prod_05 + 
			                 row.Ind_Prod_06 + row.Ind_Prod_07 + row.Ind_Prod_08 + row.Ind_Prod_09 + row.Ind_Prod_10 +
			                 row.Ind_Prod_11 + row.Ind_Prod_12 + row.Ind_Prod_13 + row.Ind_Prod_14 + row.Ind_Prod_15 +
			                 row.Ind_Prod_16 + row.Ind_Prod_17 + row.Ind_Prod_18 + row.Ind_Prod_19 + row.Ind_Prod_20 + 
			                 row.Ind_Prod_21 + row.Ind_Prod_22 + row.Ind_Prod_23 + row.Ind_Prod_24 , axis=1)
	#Eliminamos las variables Ind_Prod ya que determinamos que no son relevantes independientemente (ver informe)
	for i in range(2,10):
		del df["Ind_Prod_0"+str(i)]

	for i in range(10,25):
		del df["Ind_Prod_"+str(i)]
	
	#Transformamos los valores de Socio_Demo_01 asignando a cada individuo a un grupo obtenido anteriormente utilizando
	# la funcion make_median_clust
	df['Socio_Demo_01'] = df['Socio_Demo_01'].apply(lambda x: clust_dictionatry[x])

	#Seleccionamos los atributos que consideremos relevantes (ver informe) 
	#para posteriormente usarlos para entrenar y clasificar, si se trata del dataset de test
	# no se incluye la variable poder adquisitivo y se devuelve el indice del cliente
	if not is_test:
		return df.loc[:,['Imp_Cons_01', 'Imp_Cons_02', 'Imp_Cons_03', 'Imp_Cons_04', 
		'Imp_Cons_06', 'Imp_Cons_08', 'Imp_Cons_09', 'Imp_Cons_11', 'Imp_Cons_12', 
		'Imp_Cons_15', 'Imp_Cons_16', 'Imp_Cons_17', 'Imp_Sal_08', 'Imp_Sal_09', 
		'Imp_Sal_10', 'Imp_Sal_12', 'Imp_Sal_15', 'Imp_Sal_19', 'Imp_Sal_20', 'Imp_Sal_21', 
		'Ind_Prod_01', 'Num_Oper_05', 'Num_Oper_06', 'Num_Oper_09', 'Num_Oper_17', 'Num_Oper_18', 
		'Socio_Demo_01', 'Socio_Demo_02', 'Socio_Demo_03',"Poder_Adquisitivo"]], None
	else: 
		return df.loc[:,['Imp_Cons_01', 'Imp_Cons_02', 'Imp_Cons_03', 'Imp_Cons_04', 
		'Imp_Cons_06', 'Imp_Cons_08', 'Imp_Cons_09', 'Imp_Cons_11', 'Imp_Cons_12', 
		'Imp_Cons_15', 'Imp_Cons_16', 'Imp_Cons_17', 'Imp_Sal_08', 'Imp_Sal_09', 
		'Imp_Sal_10', 'Imp_Sal_12', 'Imp_Sal_15', 'Imp_Sal_19', 'Imp_Sal_20', 'Imp_Sal_21', 
		'Ind_Prod_01', 'Num_Oper_05', 'Num_Oper_06', 'Num_Oper_09', 'Num_Oper_17', 'Num_Oper_18', 
		'Socio_Demo_01', 'Socio_Demo_02', 'Socio_Demo_03']], client_index
		


"""
Funcion principal que entrena el modelo a partir del dataset proporcionado para entrenamiento
y clasifica los elementos que se proporcionan en test.
"""
def cajamar_prediction():

	#Carga de delos dataframes de test y train
	df1, _ = use_fram(path = "Dataset_Salesforce_Predictive_Modelling_TRAIN.txt", is_test = False)
	df_test, id_customer = use_fram(path = "Dataset_Salesforce_Predictive_Modelling_TEST.txt", is_test = True)


	#Transformacion de los dataframes de pandas a numpy arrays, en el caso de los datos de train
	# separamos el valor a predecir (poder adquisitivo) del resto de variables
	data = df1.values
	y_train = data[:,-1]
	X_train = data[:,0:-1]

	X_test = df_test.values


	#Se fijan los parametros del modelo obtenidos por validacion simple (ver informe)
	param = {'num_leaves':512, 
			'objective':'mape',
			'max_depth':10,
			'learning_rate':0.01,
			'max_bin':255,
			"min_child_samples":30,
			"min_child_weight":0.0001, 
			"boosting":"gbdt",
			"top_rate":0.5,
			"other_rate":0.1,
			"top_k":100,
			"bin_construct_sample_cnt":300000}
	param['metric'] = ["mape",'mae']

	#Se realiza una estandarizacion robusta usando mediana y rango intercuartilico (preprocesamiento de los datos)
	scalator = RobustScaler()
	scalator.fit(X_train)
	S_demo1 = deepcopy(X_train[:,26])
	test_demo1 = deepcopy(X_test[:,26])
	X_train = scalator.transform(X_train)
	X_test = scalator.transform(X_test)
	X_train[:,26] = S_demo1
	X_test[:,26] = test_demo1


	train_data=lgb.Dataset(X_train,label=y_train)

	#Se fijan el numero de iteraciones del modelo
	num_round = 1300

	#Se entrena el modelo (Light GBM)
	clf2=lgb.train(param,train_data,num_round,categorical_feature=[26])

	#Se obtenien las predicciones del modelo
	predictions=clf2.predict(X_test)

	#Se escriben las IDs de los clientes con la estimacion de su poder absoluto
	fout = open("Test_Mission.txt","w")
	fout.write("ID_Customer,PA_Est\n")

	for i in range(len(predictions)):
		fout.write(str(id_customer[i])+","+str(predictions[i])+"\n")
	fout.close()


if __name__ == "__main__":
	cajamar_prediction()