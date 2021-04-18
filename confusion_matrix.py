import utils
import Parameters
import model
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    _, parameters = Parameters.load_last()
    images , labels  = utils.load("tdataset/prepared/dev")
    
    # Prepare to train
    flatten = images.reshape(images.shape[0], -1).T
    images = flatten/255.

    preds, acc = model.predict(images, labels, parameters)

    TOTAL = labels.shape[1]

    mask = labels == 0
    tlabel = labels[mask]
    tpreds = preds[mask]

    TN = np.sum(tlabel == tpreds)
    FN = np.sum(tlabel != tpreds)

    mask = labels == 1
    tlabel = labels[mask]
    tpreds = preds[mask]

    TP = np.sum(tlabel == tpreds)
    FP = np.sum(tlabel != tpreds)

    print("TN[0-0]", TN)
    print("FN[0-1]", FN)

    print("TP[1-1]", TP)
    print("FP[1-0]", FP)

    
    # Doğruluk (Accuracy)
    '''
    Doğru sınıflandırmanın toplama bölümüdür. Yani; doğrular / toplam. Yani yoka yok vara var dediklerimizin toplama oranı. Ayrıca esas köşegenin toplama oranı da diyebiliriz.
    '''
    ACC = (TN + TP) / TOTAL
    print("Accuracy", ACC)
    
    # Precision:
    '''
    Doğru var olarak tahmin edilenlerin, toplam var tahminlere oranı.
    '''
    Precision = TP/(FP+TP)
    print("Precision", Precision)

    # Recall (Geri Çağırma)
    Recall = TP / (TP + FN)
    print("Recall", Recall)



