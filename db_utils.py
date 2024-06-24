import yaml
from sqlalchemy import create_engine
import pandas as pd

def LoadCredentials(): #function to load the RDS credentials into a dictionary
    with open('credentials.yaml','r') as f:
        cred_details = yaml.full_load(f)
        #print(cred_details)
    return cred_details

class RDSDatabaseConnector:

    def __init__(self,cred_details_dict):
        for i in cred_details_dict:
            setattr(self,i,cred_details_dict[i])
            #print(self.RDS_HOST)
        return
    
    def get_connection(self):
        engine_conection = create_engine(url="postgresql://{0}:{1}@{2}:{3}/{4}".format(
            self.RDS_USER, self.RDS_PASSWORD, self.RDS_HOST, self.RDS_PORT, self.RDS_DATABASE))
        return engine_conection                                      
        
    def read_data(self,engine_connection):
        with engine_connection.connect() as conn, conn.begin():
            data = pd.read_sql_table("loan_payments", conn)

                   
        return data
    
    def data_to_csv(self,data):
        #print(data.shape)
        data.to_csv('loan_payments.csv')
        return

    def read_data_from_csv(self):
        df = pd.read_csv('loan_payments.csv')
        #print(df.shape)
        return 
    

if __name__ == "__main__":


    cred_details_dict = LoadCredentials()
    #print(f'function return: {cred_details_dict}')

    RDSConnectors = RDSDatabaseConnector(cred_details_dict)

    try:
        engine_connection = RDSConnectors.get_connection()
<<<<<<< HEAD
        print("connection to  successful")

    except:
        print("connection not successful")
=======
        print("connection to successful")

    except:
        print("not successful")
>>>>>>> 7af197e12ed6991ecad2e839d79cade0fe2b7fe0

    data_frame = RDSConnectors.read_data(engine_connection) 

    #print(size_of_data) 

    RDSConnectors.data_to_csv(data_frame) 
    RDSConnectors.read_data_from_csv()



    


