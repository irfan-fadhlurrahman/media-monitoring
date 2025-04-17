import pytz
import time
import random
import os
import re
import time
import pandas as pd
import numpy as np
import duckdb

from GoogleNews import GoogleNews
from googlenewsdecoder import gnewsdecoder
from google.colab import userdata
from trafilatura import fetch_url, extract

from minio import Minio
from io import BytesIO
from datetime import datetime, timedelta

from obs import (
    ObsClient, PutObjectHeader,
    SetObjectMetadataHeader, HeadPermission
)

STORAGE_OPTIONS = {
    "key": userdata.get('MINIO_ACCESS_KEY'),
    "secret": userdata.get('MINIO_SECRET_KEY'),
    "use_ssl": False,
    "client_kwargs": {
        "endpoint_url": f"http://{userdata.get('MINIO_ENDPOINT')}"
    }
}

### RAW DATA OBS
def get_raw_data_obs():
    exc = obs_factory().read_object("news/exclude_media/media.parquet")
    exclude_source = ",".join([
        f"'{str(r).lower()}'" for r in exc["news_source"]
    ])

    df = obs_factory().read_object("news/backup/news_url_null.parquet")
    df = duckdb.execute(
        f"""
        select id, keyword, title, link
        from df
        where lower(source) not in ({exclude_source})
        """
    ).df()

    return df

def append_obs(df, object_name, col='news_url'):
    try:
        exist = obs_factory().read_object(object_name)
        df = pd.concat([df, exist]).reset_index(drop=True)
        df = (
            df[~(df[col].isnull())]
            .reset_index(drop=True)
            .sort_values(by=[col, 'date_modified'], ascending=False)
            .drop_duplicates(subset=[col], keep='first')
            .reset_index(drop=True)
        )
    except Exception:
        print(f'[{get_current_date()}] No existing data')
    
    return df

def get_news_url_per_keyword(df, keyword):
    df_k = df[df['keyword'] == keyword].sample(frac=1).reset_index(drop=True)
    if df_k.empty:
        print(f'No data for {keyword} exists')
        return None
    
    df_k['news_url'] = df_k['link'].apply(decode_google_news_link)
    df_k['date_modified'] = get_current_date()

    t = get_current_date().strftime('%Y%m%d')
    filename = f"/content/drive/MyDrive/Full News/chunks_duckdb/{keyword}_{t}.parquet"
    df_k.to_parquet(filename, index=False)

    object_name=f"news/chunks_duckdb/{keyword}_{t}.parquet"
    df_k = append_obs(df_k, object_name)
    obs_factory().write_object(df_k, object_name)

    return df_k
    

### DATAFRAME CHUNKER
def chunk_dataframe_numpy(df, num_chunks):
    np_array = df.values
    chunks_np = np.array_split(np_array, num_chunks)
    for chunk_np in chunks_np:
        yield pd.DataFrame(chunk_np, columns=df.columns)

### GOOGLE NEWS FULL TEXT
def get_full_text(url):
    try:
        return extract(
            fetch_url(url),
            favor_precision=True,
        )
    except Exception:
        return None

### GOOGLE NEWS URL DECODER
def decode_google_news_link(url):
    try:
        decoded_url = gnewsdecoder(url, interval=5)
        return decoded_url["decoded_url"]
    except Exception:
        return None

### GOOGLE NEWS EXTRACTOR
def extract_google_news(keyword, selected_date, lang="id", is_print=True):
    start = selected_date.strftime('%m/%d/%Y')
    end = (selected_date + timedelta(days=1)).strftime('%m/%d/%Y')

    googlenews = GoogleNews(lang=lang)
    googlenews.set_time_range(start, end)
    googlenews.set_encode('utf-8')
    googlenews.get_news(keyword)

    df = pd.DataFrame(googlenews.results(sort=True))
    if df.empty:
        if is_print:
            print(selected_date, 'FAILED')
        return pd.DataFrame()

    df['date_created'] = get_current_date()
    df['date_modified'] = get_current_date()
    df['start_date'] = selected_date
    df['end_date'] = selected_date + timedelta(days=1)
    df['keyword'] = keyword

    googlenews.clear()
    del googlenews

    if is_print:
        print(selected_date, 'SUCCESS', len(df), 'rows')
    
    return df

def load_to_drive(df_list, keyword):
    df_list = [df for df in df_list if not df.empty]
    if df_list:
        df = (
            pd.concat(df_list, ignore_index=True).reset_index(drop=True)
            .sort_values(by=['date_created'], ascending=False)
            .drop_duplicates(subset=['title', 'media'], keep='first')
            .reset_index(drop=True)
        )
        filename = f"/content/drive/MyDrive/News/{keyword}_{get_current_date().strftime('%Y%m%d')}.parquet"
        df.to_parquet(filename, index=False)
        print(f"Saved to {filename}")
        print(f"Total rows: {len(df)}\n")

def run_gnews_extractor(keyword, start_date, end_date):
    df_list = []
    for selected_date in pd.date_range(start_date, end_date, freq='D')[::-1]:
        df = extract_google_news(keyword=keyword, selected_date=selected_date)
        if len(df) > 0:
            df_list.append(df)
            tmp = load_to_drive(df_list=df_list, keyword=keyword)


### DATETIME
def get_current_date(days_diff: int = 0):
    return (datetime.now(tz=pytz.timezone('Asia/Jakarta')) - timedelta(days=days_diff)) \
        .replace(tzinfo=None, microsecond=0)

### OBS
class OBSHuawei:
    def __init__(self, credentials: dict, is_print=True):
        self._access_key = credentials['access_key']
        self._secret_key = credentials['secret_key']
        self._server = credentials['server']
        self.bucket_name = credentials['bucket_name']
        self.read = {
            "csv": pd.read_csv,
            "parquet": pd.read_parquet,
            "pq": pd.read_parquet,
            "json": pd.read_json,
        }
        self.write = lambda df: {
            "csv": df.to_csv,
            "parquet": df.to_parquet,
            "pq": df.to_parquet,
            "json": df.to_json,
        }
        self.is_print = is_print
    
    def client(self):
        self.obs = ObsClient(
            access_key_id=self._access_key,    
            secret_access_key=self._secret_key,
            server=self._server, is_secure=False
        )
        return self.obs
    
    def _print_response(self, resp):
        if not self.is_print:
            return None
        if resp.status < 300:
            print('Put Content Succeeded')
            print('requestId:', resp.requestId)
        else:
            print('Put Content Failed')
            print('requestId:', resp.requestId)
            print('errorCode:', resp.errorCode)
            print('errorMessage:', resp.errorMessage)
    
    def _set_public(self, nama_file_obs):
        aclControl = HeadPermission.PUBLIC_READ
        self.obs.setObjectAcl(self.bucket_name, nama_file_obs, aclControl=aclControl)
        print(f"https://{self.bucket_name}.{self._server}/{nama_file_obs}")   


    def write_object(self, obj, object_path):
        obj_format = object_path.split('.')[-1]
        ext = list(self.read.keys())+['pkl','pickle']
        metadata = {'ACL': 'public-read', 'author': 'databoks'}
        headers = SetObjectMetadataHeader()

        if obj_format not in ext:
            print(f'Currently not support {obj_format}')
            return None
        
        if obj_format == 'csv':
            obj_bytes = self.write(obj)[obj_format]().encode('utf-8')
            datatype = 'text/plain'
        elif (obj_format == 'pkl') or (obj_format == 'pickle'):
            obj_bytes = pickle.dumps(obj)
            datatype = 'text/plain'
        else:
            obj_bytes = self.write(obj)[obj_format]()
            datatype = 'text/plain'
            
        self.client()
        resp = self.obs.putContent(
            bucketName=self.bucket_name, 
            objectKey=object_path, 
            content=BytesIO(obj_bytes),
            metadata= metadata,
            headers=PutObjectHeader(
                acl='public-read',
                contentLength=len(obj_bytes), contentType= datatype
            )
        )
        self._print_response(resp)  
        self._set_public(object_path)
    
    def read_object(self, object_path):
        obj_format = object_path.split('.')[-1]
        ext = list(self.read.keys())
        if obj_format not in ext:
            print(f'Currently not support {obj_format}')
            return None
        
        self.client()
        bucket_client = self.client().bucketClient(self.bucket_name)
        resp = bucket_client.getObject(
            object_path, 
            loadStreamInMemory=True
        )
        if (obj_format == 'pkl') or (obj_format == 'pickle'):
            return pickle.loads(resp.body.buffer)
        else:
            obj_read = self.read[obj_format]
            binary_object = BytesIO(resp.body.buffer)
            binary_object.seek(0)
            
            return obj_read(binary_object)

### OBS SHORTCUT
def obs_factory(is_print=False):
    credentials = {
        "access_key": userdata.get('OBS_ACCESS_KEY'),
        "secret_key": userdata.get('OBS_SECRET_KEY'),
        "server": userdata.get('OBS_SERVER'),
        "bucket_name": "data-team"
    }
    return OBSHuawei(credentials, is_print)

### MINIO
def read_parquet_minio(object_name):
    return pd.read_parquet(
        f"s3://data-lake/{object_name}", 
        storage_options=STORAGE_OPTIONS
    )

def save_to_minio(df, object_name):
    client = Minio(
        userdata.get('MINIO_ENDPOINT'),
        access_key=userdata.get('MINIO_ACCESS_KEY'),
        secret_key=userdata.get('MINIO_SECRET_KEY'),
        secure=False,
    )
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    client.put_object(
        bucket_name=userdata.get('MINIO_BUCKET_NAME'),
        object_name=object_name,
        data=buffer,
        length=buffer.getbuffer().nbytes,
    )
    print(f"Saved to s3://{userdata.get('MINIO_BUCKET_NAME')}/{object_name}")






















































































