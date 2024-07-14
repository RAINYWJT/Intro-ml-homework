from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
 
y_label = ([0,1,0,1,0,1,0,0])  
y_pre = ([0.32,0.89,0.63,0.32,0.25,0.66,0.48,0.8])
fpr, tpr, thersholds = roc_curve(y_label, y_pre, pos_label=1)
 
for i, value in enumerate(thersholds):
    print("%f %f %f" % (fpr[i], tpr[i], value))
 
roc_auc = auc(fpr, tpr)
 
plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
 
plt.xlim([-0.05, 1.05]) 
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig("roc_drawing.png")
plt.show()


print("AUC: %.2f" % roc_auc)