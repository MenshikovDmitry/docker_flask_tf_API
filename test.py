import os

try:
    from app import app
    import unittest
except Exception as e:
    print("Import Error",e)

PAGES=[
        "/",
        "/test",
        "/log",
        "/predict/istruck",
        "/v",


    ]

from models.truck_detector import Predictor
from keras.preprocessing import image






class ModelTest(unittest.TestCase):

    def test_trucks(self):
        predictor=Predictor()
        folder="test/truck"
        expected_label="TRUCK"

        files=os.listdir(folder)
        for filename in files:
            img = image.load_img(os.path.join(folder,filename), target_size=predictor.input_size)
            result=predictor.predict(img)
            self.assertEqual(result["prediction"],expected_label,
                msg="Wrong model prediction. file: {}. \nExpected {}, got:{}".format(filename,expected_label,result))

        del(predictor)

    def test_not_trucks(self):
        predictor=Predictor()
        folder="test/not_truck"
        expected_label="not truck"

        files=os.listdir(folder)
        for filename in files:
            img = image.load_img(os.path.join(folder,filename), target_size=predictor.input_size)
            result=predictor.predict(img)
            self.assertEqual(result["prediction"],expected_label,
                msg="Wrong model prediction. file: {}. \nExpected {}, got:{}".format(filename,expected_label,result))

        del(predictor)


class FlaskTest(unittest.TestCase):
    
    def test_200(self):
        tester=app.test_client(self)
        for request in PAGES:
            #print("requesting",request)
            response=tester.get(request)
            statuscode=response.status_code
            self.assertEqual(statuscode,200,msg=request)


if __name__=="__main__":
    unittest.main()