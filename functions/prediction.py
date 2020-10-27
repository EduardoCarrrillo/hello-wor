import os
import json
import joblib
import pandas as pd
import numpy as np
import pymongo
import dns
import ssl
from scipy import sparse
from scipy.spatial.distance import hamming
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity
global client, prods, mydb,colProds,tiendas
#client = pymongo.MongoClient("mongodb+srv://Team-Axity2020:Axity2020@cluster0.n3b8v.mongodb.net",ssl=True,ssl_cert_reqs='CERT_NONE')
#mydb = client["POC-lacomer"]
#colProds = mydb["productos"]
#colClients=mydb['clientes']
#colTransactions = mydb['transactions']
uri = "mongodb://poxdemo:bWRuF3PVqA1CjlDzRzf0KhZPnPBK55S6p89kyFXxzQpzmGhEhopwHk8Z4lPaVlQHxUIjA30ypoTBR5cQtrkWYg==@poxdemo.mongo.cosmos.azure.com:10255/POC-lacomer?ssl=true&retrywrites=false&replicaSet=globaldb&maxIdleTimeMS=120000&appName=@poxdemo@"
clientAzure = pymongo.MongoClient(uri,ssl_cert_reqs=ssl.CERT_NONE)
mydb = clientAzure["POC-lacomer"]
colProds = mydb["products"]
colClients=mydb['clients']
colTrans=mydb['transactions']
tiendas = {"city":363,"sumesa":233,"lacomer":287,"fresko":137}
class predict():
    
    @classmethod
    def distance2(user1,user2):
        client1_matrix = matrix.transpose()[user1]
        client2_matrix = matrix.transpose()[user2]
        return hamming(client1_matrix,client2_matrix)
    
    @classmethod
    def get_promociones(cls):
        #skus_proms = [8411320282861,7312040090754,830809007043,192563942788,7501950007123,53891136402,7501059295889]
        #skus_proms=[7501035012035,7501102614322,8411320282861]
        skus_proms=[7702073001603,8806086896771,99053013211,4548736057685,2610833000009,8411320282861,7312040090754,830809007043,192563942788,7501950007123,53891136402,7501059295889]
        skus = list(colProds.find({'id': {"$in": skus_proms}}))
        df = pd.DataFrame(skus).head(10)[['id','name','url','price']]
        dic = df.to_dict('records')
        return dic
    
    @classmethod
    def get_items(cls):
        #skus_apriori = [7501048840212,7501027300201,7501000922352,7502272870198,7502249130713]
        skus_apriori=[7502236172054,7791293033198,5000264012790,5011013100156,7500327047878,7501048840212,7501027300201,7501000922352,7502272870198,7502249130713]
        skus = list(colProds.find({'id': {"$in": skus_apriori}}))
        df = pd.DataFrame(skus).head(10)[['id','name','url','price']]
        dic = df.to_dict('records')
        return dic
    
    @classmethod
    def user_knn(cls,usuario,store):
        def distance(user1,user2):
            client1_matrix = matrix.transpose()[user1]
            client2_matrix = matrix.transpose()[user2]
            return hamming(client1_matrix,client2_matrix)
        transactions_2 = pd.read_csv("./modelos/knn_{}.csv".format(tiendas[store]))
        client_per_sku = transactions_2.sku.value_counts()
        sku_per_client = transactions_2.client_id.value_counts()
        transactions_3 = transactions_2[transactions_2.sku.isin(client_per_sku[client_per_sku>30].index)]
        transactions_4 = transactions_3[transactions_3.client_id.isin(sku_per_client[sku_per_client>30].index)]
        matrix = pd.pivot_table(transactions_4, values="quantity", index=['client_id'], columns=['sku'])
        allUsers = pd.DataFrame(matrix.index)
        print(allUsers.shape)
        allUsers = allUsers[allUsers.client_id!=usuario]
        print(allUsers.shape)
        allUsers["distance"] = allUsers["client_id"].apply(lambda x: distance(usuario,x))
        K=10
        KallUsers = allUsers.sort_values(["distance"], ascending=False)["client_id"][:K]
        NNRating = matrix[matrix.index.isin(KallUsers)]
        avgRating = NNRating.apply(np.nanmean).dropna()
        items_ya = matrix.transpose()[usuario].dropna().index
        skus=avgRating[~avgRating.index.isin(items_ya)].sort_values(ascending=False)[:10].index.tolist()
        query = {'id': {"$in": skus}}
        df_products = list(colProds.find(query))
        products = pd.DataFrame(df_products)
        products = products[['id','name','url','price']]
        dic = products.to_dict('records')
        return dic
    
    @classmethod
    def user(cls,usuario,store):
        tienda = tiendas[store]
        print(store)
        print(tienda)
        print("Recomendando por usuario")
        model = joblib.load('./modelos/model_store_{}.pkl'.format(tienda))
        products_features = sparse.load_npz("./modelos/products_features_{}.npz".format(tienda))
        clients_features = sparse.load_npz("./modelos/clients_features_{}.npz".format(tienda))
        #data = request.get_json(force=True)
        user = usuario
        print(user)
        query ={"client_id":int(user)}
        doc = colClients.find(query)
        doc_list = list(doc)
        client_id_num = doc_list[0]['client_id_num']
        print (client_id_num)
        doc1 = list(colTransactions.find(query))
        skus_previos = list(map(lambda x: x['sku'], doc1))
        query2 = {'id': {"$nin": skus_previos}}
        skus_previos_ids = list(colProds.find(query2))
        df_previous = pd.DataFrame(skus_previos_ids)
        skus_to_predict = list(map(lambda x: x['product_id_num'],skus_previos_ids))
        scores = model.predict(
                client_id_num,
                skus_to_predict,
                item_features=products_features,
                user_features=clients_features)
        df_previous['scores'] = scores
        df_previous = df_previous[df_previous.department!="  FRUTAS Y VERDURAS"].reset_index(drop=True)
        df_previous = df_previous[df_previous.name!="POR CLASIFICAR"].reset_index(drop=True)
        df_to_recommended = df_previous.sort_values(by='scores', ascending=False)[:10][['id','name','url','price']]
        dic = df_to_recommended.to_dict('records')
        
        return dic

    @classmethod
    def user_products(cls,usuario,store):
        tienda = tiendas[store]
        print(store)
        print(tienda)
        print("Recomendando por usuario y sus productos")
        #data = request.get_json(force=True)
        user = usuario
        print(user)
        transactions=pd.read_csv("./modelos/transactions_{}.csv".format(tiendas[store]))
        #query ={"store_id":int(store)}
        #transactions=pd.DataFrame(list(colTrans.find(query)))
        transactions2=transactions[transactions.client_id==user][['sku','quantity']]
        transactions3=transactions2.groupby("sku")['quantity'].agg('sum').reset_index().sort_values(by="quantity", ascending=False).head(10)
        skus = transactions3.sku.values.tolist()
        query2 = {'id': {"$in": skus}}
        skus_previos_ids = list(colProds.find(query2))
        df_previous = pd.DataFrame(skus_previos_ids)
        df_previous = df_previous[df_previous.name!="POR CLASIFICAR"].reset_index(drop=True)
        df=df_previous[['id','name','url','price']]
        dic = df.to_dict('records')
        return dic
    
    
    @classmethod
    def similar_items(cls,itemm,store):
        tienda=tiendas[store]
        print("Recomendando por item")
        print(store)
        print(tienda)
        print(itemm)
        model = joblib.load('./modelos/model_store_{}.pkl'.format(tienda))
        products_features = sparse.load_npz("./modelos/products_features_{}.npz".format(tienda))
        clients_features = sparse.load_npz("./modelos/clients_features_{}.npz".format(tienda))
        item = itemm
        query = {'id': int(item)}
        prod = list(colProds.find(query))
        prod_id_num = prod[0]['product_id_num']
        print(prod_id_num)
        item_biases, item_embeddings = model.get_item_representations(products_features)
        scores = cosine_similarity(item_embeddings,item_embeddings[prod_id_num].reshape(1, -1))
        from operator import itemgetter
        result_lst = [(i,scores[x]) for i,x in enumerate(range(scores.shape[0]))]
        result_lst_srt = sorted(result_lst, key=itemgetter(1), reverse=True)
        ty = list(map(lambda x:x[0], result_lst_srt[:11]))
        scores = list(map(lambda x:x[1][0], result_lst_srt[:11]))
        a = dict(map(lambda x,y: (x, y), ty, scores))
        query = {'product_id_num': {"$in": ty}}
        df_products = list(colProds.find(query))
        products = pd.DataFrame(df_products)
        products['scores'] = products.apply(lambda x: float(a[int(x['product_id_num'])]), axis=1)
        products = products.sort_values(by='scores', ascending=False)
        products = products[products.id!=item].reset_index(drop=True)
        products = products[['id','name','url','price']]
        dic = products.to_dict('records')
        return dic

    @classmethod
    def similar_items2(cls,itemm,store):
        tienda=tiendas[store]
        print("Recomendando por item")
        print(store)
        print(tienda)
        print(itemm)
        model = joblib.load('./modelos/model_store_{}.pkl'.format(tienda))
        products_features = sparse.load_npz("./modelos/products_features_{}.npz".format(tienda))
        clients_features = sparse.load_npz("./modelos/clients_features_{}.npz".format(tienda))
        item = itemm
        query = {'id': int(item)}
        prod = list(colProds.find(query))
        prod_id_num = prod[0]['product_id_num']
        print(prod_id_num)
        item_biases, item_embeddings = model.get_item_representations(products_features)
        scores = cosine_similarity(item_embeddings,item_embeddings[prod_id_num].reshape(1, -1))
        from operator import itemgetter
        result_lst = [(i,scores[x]) for i,x in enumerate(range(scores.shape[0]))]
        result_lst_srt = sorted(result_lst, key=itemgetter(1), reverse=True)
        ty = list(map(lambda x:x[0], result_lst_srt[:30]))
        scores = list(map(lambda x:x[1][0], result_lst_srt[:30]))
        a = dict(map(lambda x,y: (x, y), ty, scores))
        query = {'product_id_num': {"$in": ty}}
        df_products = list(colProds.find(query))
        products = pd.DataFrame(df_products)
        products['scores'] = products.apply(lambda x: float(a[int(x['product_id_num'])]), axis=1)
        products = products.sort_values(by='scores', ascending=False)
        products = products[products.id!=item].reset_index(drop=True)
        products = products[['id','name','url','price']].tail(9)
        products = products[products.name!="POR CLASIFICAR"].reset_index(drop=True)
        dic = products.to_dict('records')
        return dic
    
    @classmethod
    def apriori_model(cls,item,store):
        tienda=tiendas[store]
        df = pd.read_csv("./modelos/apriori_{}.csv".format(tienda))
        prods = df[df.item_A==int(item)].reset_index(drop=True)[['item_B','lift']].to_dict('records')
        df_apriori = pd.DataFrame(prods)
        print(df_apriori.shape)
        df_apriori.rename({'lift':'score','item_B':'id'}, axis=1, inplace=True)
        list_prods = list(map(lambda x: x['item_B'], prods))
        query = {'id': {"$in": list_prods}}
        df_products = list(colProds.find(query))
        products = pd.DataFrame(df_products)[['id','name','url','price']]
        products = products[products.name!="POR CLASIFICAR"].reset_index(drop=True)
        products2 = products.merge(df_apriori, on="id", how="inner")
        products2=products2.head(10)
        dic = products2.to_dict('records')
        print(dic)
        return dic
