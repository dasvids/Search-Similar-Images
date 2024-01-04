from imports import *

class BaseImageProcessor():
    def __init__(self,db_path:str):
        self.__db_load__(db_path)
        self.sift = cv.SIFT_create()
        self.kmeans = load('kmeans.joblib')

    def __db_load__(self,db_path:str):
        self.db = pd.read_csv(db_path,delimiter='\t')
        self.db['encoding_vector'] = self.db['encoding_vector'].apply(lambda x: eval(x))

    def __vectorize_image__(self,image:np.array,n_clusters=512):
        _,des = self.sift.detectAndCompute(image,None)
        classes = self.kmeans.predict(des)
        hist = np.zeros(n_clusters)
        for clss in classes:
            hist[clss] += 1

        hist /= len(classes)
        return hist

    def get_image(self,ind:int):
        if ind < 0 or ind > len(self.db):
            return np.zeros((300,300))
        return cv.imread(self.db.loc[ind]['filepath'])

    @abstractmethod
    def find_n_similar(self,image:np.array,n_similar=5):
        pass
    

class KNNImageProcessor(BaseImageProcessor):
    def __init__(self,db_path):
        super().__init__(db_path)
        self.nbrs = load('nbrs.joblib')

    def find_n_similar(self,image:np.array,n_similar=5):
        img_vec = super().__vectorize_image__(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
        _,inds = self.nbrs.kneighbors([img_vec],n_neighbors=n_similar)
        return inds.flatten()
    
def run():
    processor = KNNImageProcessor('db.csv')

    st.title('Search Similar Images')
    file = st.file_uploader('Upload Image')

    if file:
        img = cv.imdecode(np.frombuffer(file.getbuffer(), dtype=np.uint8), cv.IMREAD_COLOR)
        st.image(img, channels='BGR', caption='Uploaded Image')

        st.title('Similar Images')
        
        similar_images = [cv.cvtColor(processor.get_image(ind), cv.COLOR_BGR2RGB) for ind in processor.find_n_similar(img)]
        st.image(similar_images, caption=[f'Similar Image #{ind + 1}' for ind in processor.find_n_similar(img)])


if __name__ == '__main__':
    run()