import csv
import tkMessageBox
from Tkinter import *
import tkSimpleDialog
from tkFileDialog import askopenfilename, asksaveasfile
import sklearn
from sklearn import linear_model
from sklearn.cross_validation import cross_val_predict
import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.stats import pearsonr

root = Tk()
root.title("Predicting Glycaemic Index")
root.geometry('1200x720')
root.resizable(width=False, height=True)
global lb
global switch
switch = 0

# set reactions for a single click on the list box
def print_item(event):
    global current_row
    global lb
    cs = lb.get(lb.curselection())
    set_notice.set('')
    for row in foodlist:
        if cs == row['Item']:
            current_row = row
            FoodName.set(row['Item'])
            GI.set(row['GI(Glucose= 100)'])
            STD.set(row['Standard Deviation'])
            Subjects.set(row['Subjects (type & number)'])
            Water.set(row['Water_(g)'])
            Energy.set(row['Energ_Kcal'])
            Protein.set(row['Protein_(g)'])
            Fat.set(row['Lipid_Tot_(g)'])
            Carbohydrate.set(row['Carbohydrt_(g)'])
            Fibre.set(row['Fiber_TD_(g)'])
            Sugar.set(row['Sugar_Tot_(g)'])
            USDA_foodname.set(row['Matching_Food'])

# define two states of the list box: select single and select multiple
def listbox():
    global lb
    if switch == 0:
        lb = Listbox(root, height=35, width=75, selectmode=SINGLE)
        lb.bind('<ButtonRelease-1>', print_item)
        lb.grid(row=1, column=0, columnspan=5, rowspan=80, sticky=W + E + N + S, padx=20, pady=15)

        # create a vertical scrollbar to the right of the listbox
        yscroll = Scrollbar(command=lb.yview, orient=VERTICAL)
        yscroll.grid(row=1, column=1, columnspan=4, rowspan=80, sticky=N + S + E, padx=20, pady=15)
        lb.configure(yscrollcommand=yscroll.set)
        if 'foodlist' in globals():
           for row in foodlist:
               item = row['Item']
               lb.insert(END, item)
    else:
       lb = Listbox(root, height=35, width=75, selectmode=MULTIPLE)
       lb.bind('<ButtonRelease-1>', get_multi)
       lb.grid(row=1, column=0, columnspan=5, rowspan=80, sticky=W + E + N + S, padx=20, pady=15)

       # create a vertical scrollbar to the right of the listbox
       yscroll = Scrollbar(command=lb.yview, orient=VERTICAL)
       yscroll.grid(row=1, column=1, columnspan=4, rowspan=80, sticky=N + S + E, padx=20, pady=15)
       lb.configure(yscrollcommand=yscroll.set)
       for row in foodlist:
           item = row['Item']
           lb.insert(END, item)

# invoke the list box at beggining
listbox()

# load csv data by asking a file name
def load_csv():
    global lb
    lb.delete(0, END)
    Tk().withdraw()
    global load_filename
    global foodlist
    load_filename = askopenfilename()
    reader = csv.DictReader(open(load_filename, 'rU'))
    foodlist = list(reader)
    for row in foodlist:
        item = row['Item']
        lb.insert(END, item)

# save csv data with a defalut name
def save():
    FIELDS = ['Food Number', 'Item', 'GI(Glucose= 100)', 'Standard Deviation', 'Subjects (type & number)', 'Water_(g)',
              'Energ_Kcal',
              'Protein_(g)', 'Lipid_Tot_(g)', 'Carbohydrt_(g)', 'Fiber_TD_(g)', 'Sugar_Tot_(g)', 'Matching_Food',
              'Similarity']
    writer = csv.DictWriter(load_filename, fieldnames=FIELDS)
    writer.writerow(dict(zip(FIELDS, FIELDS)))
    for row in foodlist:
        writer.writerow(row)

# save csv data by asking a file name
def save_csv():
    Tk().withdraw()
    filename = asksaveasfile(mode='w', defaultextension=".csv")
    print(filename)
    if filename is None:  # asksaveasfile return `None` if dialog closed with "cancel".
        return
    FIELDS = ['Food Number', 'Item', 'GI(Glucose= 100)', 'Standard Deviation', 'Subjects (type & number)', 'Water_(g)',
              'Energ_Kcal',
              'Protein_(g)', 'Lipid_Tot_(g)', 'Carbohydrt_(g)', 'Fiber_TD_(g)', 'Sugar_Tot_(g)', 'Matching_Food',
              'Similarity']
    writer = csv.DictWriter(filename, fieldnames=FIELDS)
    writer.writerow(dict(zip(FIELDS, FIELDS)))
    for row in foodlist:
        writer.writerow(row)

# change state of the list box to select multiple
def select_multiple():
    global switch
    switch = 1
    listbox()

# get the selection on select motiple state
def get_multi(event):
    global multiselected
    multiselected = lb.curselection()



# widgets on the right side
Label(root, text="Food Name: ", font=("Arial", 16)).grid(row=0, column=5, sticky=W)
FoodName = StringVar()
Entry(root, textvariable=FoodName, width=65).grid(row=1, column=5, columnspan=3)

Label(root, text="GI(Glucose= 100): ", font=("Arial", 16)).grid(row=2, column=5)
GI = StringVar()
Entry(root, textvariable=GI).grid(row=3, column=5)

Label(root, text="Standard Deviation: ", font=("Arial", 16)).grid(row=2, column=6)
STD = StringVar()
Entry(root, textvariable=STD).grid(row=3, column=6)

Label(root, text="Subjects: ", font=("Arial", 16)).grid(row=2, column=7)
Subjects = StringVar()
Entry(root, textvariable=Subjects).grid(row=3, column=7)

Label(root, text="Water: ", font=("Arial", 16)).grid(row=4, column=5)
Water = StringVar()
Entry(root, textvariable=Water).grid(row=5, column=5)

Label(root, text="Energy: ", font=("Arial", 16)).grid(row=4, column=6)
Energy = StringVar()
Entry(root, textvariable=Energy).grid(row=5, column=6)

Label(root, text="Protein: ", font=("Arial", 16)).grid(row=4, column=7)
Protein = StringVar()
Entry(root, textvariable=Protein).grid(row=5, column=7)

Label(root, text="Fat: ", font=("Arial", 16)).grid(row=6, column=5)
Fat = StringVar()
Entry(root, textvariable=Fat).grid(row=7, column=5)

Label(root, text="Carbohydrate: ", font=("Arial", 16)).grid(row=6, column=6)
Carbohydrate = StringVar()
Entry(root, textvariable=Carbohydrate).grid(row=7, column=6)

Label(root, text="Fibre: ", font=("Arial", 16)).grid(row=6, column=7)
Fibre = StringVar()
Entry(root, textvariable=Fibre).grid(row=7, column=7)

Label(root, text="Sugar: ", font=("Arial", 16)).grid(row=8, column=5)
Sugar = StringVar()
Entry(root, textvariable=Sugar).grid(row=9, column=5)


# Menu functionalities
# plot the histogram for matching scores
def histogram():
    similarity = [row['Similarity'] for row in foodlist if not row['Similarity'] == '']
    similarity = np.array(similarity).astype(np.float)
    plt.hist(similarity, bins='auto')
    plt.xlabel('Similarity')
    plt.ylabel('Number')
    plt.title(r'$\mathrm{Histogram\ of\ Similarities}$')
    plt.grid(True)
    plt.show()

# set a threshold for matching result
def set_threshold():
    global threshold
    # ask a percentage
    ask_ratio = tkSimpleDialog.askstring('Set Threshold', 'Enter the percentage of remaining data (e.g. 10%):')
    # convert a percentage to a float number
    ratio = 1 - float(ask_ratio.strip('%'))/100
    s = [row['Similarity'] for row in foodlist if row['Water_(g)'] != '-1' and row['Similarity'] != '1']
    s.sort()
    t = int(len(s) * ratio)
    threshold = s[t]
    print 'set the threshold of the highest' , ask_ratio,'of data'

# enable the user to select a data entry protocol
def set_match(m):
    global match
    match = m

# cross reference data based on a selected protocol
def  cross_refer():
    for row in foodlist:
        if row['Water_(g)'] == '-1':
            Item = row['Item'].replace(',', ' delimiter ')
            split1 = re.findall(r"[\w']+", Item)
            print '-----------Original food name----------:', row['Item']
            global max_similarity_total
            max_similarity_total = []
            max_similarity = 0
            usda = csv.DictReader(open('data/ABBREV.csv', 'rU'))
            usda_list = list(usda)
            for item in usda_list:
                usda_upper = item['Shrt_Desc'].replace("W/", "WITH ")
                usda_foodname = usda_upper.lower()
                split2 = re.findall(r"[\w']+", usda_foodname)
                if match == 1:
                    min_similarity(split1, split2)
                if match == 2:
                    min_with_weights(split1, split2)
                if match == 3:
                    min_with_hierarchy(split1, split2)
                if match == 4:
                    average_with_weights(split1, split2)
                if match == 5:
                    average_with_hierarchy(split1, split2)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_guess = usda_foodname
                    print '(', similarity, ')', best_guess
                    row['Water_(g)'] = item['Water_(g)']
                    row['Energ_Kcal'] = item['Energ_Kcal']
                    row['Protein_(g)'] = item['Protein_(g)']
                    row['Lipid_Tot_(g)'] = item['Lipid_Tot_(g)']
                    row['Carbohydrt_(g)'] = item['Carbohydrt_(g)']
                    row['Fiber_TD_(g)'] = item['Fiber_TD_(g)']
                    row['Sugar_Tot_(g)'] = item['Sugar_Tot_(g)']
            row['Matching_Food'] = best_guess
            row['Similarity'] = max_similarity
            max_similarity_total.append(max_similarity)
        else: row['Similarity'] = 1

# menu display
menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Load", command=load_csv)
filemenu.add_command(label="Save", command=save)
filemenu.add_command(label="Save as...", command=save_csv)

filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)

editmenu = Menu(menubar, tearoff=0)
editmenu.add_separator()
editmenu.add_command(label="Set Threshold", command=set_threshold)

submenu = Menu(menubar, tearoff=0)
editmenu.add_cascade(label="Set Protocol", menu=submenu)

submenu.add_radiobutton(label="Minimum Score", command=set_match(1))
submenu.add_radiobutton(label="Minimum with Decreasing", command=set_match(2))
submenu.add_radiobutton(label="Minimum with Hierarchical", command=set_match(3))
submenu.add_radiobutton(label="Average with Decreasing", command=set_match(4))
submenu.add_radiobutton(label="Average with Hierarchical", command=set_match(5))

editmenu.add_command(label="Cross-reference", command=cross_refer)

menubar.add_cascade(label="Setting", menu=editmenu)
plotmenu = Menu(menubar, tearoff=0)
plotmenu.add_command(label="Similarity Histogram", command=histogram)
menubar.add_cascade(label="Plot", menu=plotmenu)

helpmenu = Menu(menubar, tearoff=0)
menubar.add_cascade(label="Help", menu=helpmenu)

root.config(menu=menubar)

# minimal score with same weights
def min_similarity(a, b):
    global similarity
    s = 0
    for x in a:
        for y in b:
            if x != 'with' and x == y:
                s += 1
    similarity = min(float(s) / float(len(a)), float(s) / float(len(b)))

# minimal score with descreasing weights
def min_with_weights(a, b):
    global similarity
    s = 0
    for x in a:
        for y in b:
            if x != 'with' and x == y:
                s += 0.5 ** a.index(x)
    similarity = min(float(s) / float(len(a)), float(s) / float(len(b)))

# average score with descreasing weights
def average_with_weights(a, b):
    global similarity
    s = 0
    for x in a:
        for y in b:
            if x != 'with' and x == y:
                s += 0.5 ** a.index(x)
    similarity = (float(s) / float(len(a)) + float(s) / float(len(b))) / 2

# minimal score with hierarchical weights
def min_with_hierarchy(a, b):
    global similarity
    s = 0
    n = 0
    for x in a:
        if x == 'delimiter':
            n += 1
        for y in b:
            if x != 'with' and x == y:
               s += 0.5 ** n
    similarity = min(float(s) / float(len(a)), float(s) / float(len(b)))

# average score with hierarchical weights
def average_with_hierarchy(a, b):
    global similarity
    s = 0
    n = 0
    for x in a:
        if x == 'delimiter':
            n += 1
        for y in b:
            if x != 'with' and x == y:
                s += 0.5 ** n
    similarity = (float(s) / float(len(a)) + float(s) / float(len(b))) / 2

# update attributes' values for a selected food
def set():
    global Threshold
    current_row['Item'] = FoodName.get()
    current_row['GI(Glucose= 100)'] = GI.get()
    current_row['Standard Deviation'] = STD.get()
    current_row['Subjects (type & number)'] = Subjects.get()
    current_row['Water_(g)'] = Water.get()
    current_row['Energ_Kcal'] = Energy.get()
    current_row['Protein_(g)'] = Protein.get()
    current_row['Lipid_Tot_(g)'] = Fat.get()
    current_row['Carbohydrt_(g)'] = Carbohydrate.get()
    current_row['Fiber_TD_(g)'] = Fibre.get()
    current_row['Sugar_Tot_(g)'] = Sugar.get()
    row = current_row
    lb.insert(ACTIVE, row['Item'])
    s = lb.curselection()
    lb.delete(s[0])
    set_notice.set('Set!')

# add a row of new food by asking a food name
def add():
    addFoodName = tkSimpleDialog.askstring('AskFoodName', 'Enter the food name:')
    row = {'Item': addFoodName, 'GI(Glucose= 100)': '', 'Standard Deviation': '', 'Subjects (type & number)': '',
           'Water_(g)': '', 'Energ_Kcal': '', 'Protein_(g)': '', 'Lipid_Tot_(g)': '', 'Carbohydrt_(g)': '',
           'Fiber_TD_(g)': '', 'Sugar_Tot_(g)': '', 'Matching_Food': '', 'Similarity': ''}
    foodlist.append(row)
    lb.insert(0, row['Item'])

# delete the selected food
def delete():
    foodlist.remove(current_row)
    s = lb.curselection()
    lb.delete(s[0])

# train a prediction equation, predict GI for a mixed meal
def model():
    global switch
    t = {key: [] for key in 'ABCDEFGH'}
    if not 'threshold' in globals():
        set_threshold()
    # train a predicion equation, visualize the predicting result
    if switch == 0:
          for row in foodlist:
              if row['Water_(g)'] != '-1' and row['Similarity'] >= threshold:
                  if row['Water_(g)'] == '':
                     row['Water_(g)'] = '0'
                  if row['Energ_Kcal'] == '':
                     row['Energ_Kcal'] = '0'
                  if row['Protein_(g)'] == '':
                     row['Protein_(g)'] = '0'
                  if row['Lipid_Tot_(g)'] == '':
                     row['Lipid_Tot_(g)'] = '0'
                  if row['Carbohydrt_(g)'] == '':
                     row['Carbohydrt_(g)'] = '0'
                  if row['Fiber_TD_(g)'] == '':
                     row['Fiber_TD_(g)'] = '0'
                  if row['Sugar_Tot_(g)'] == '':
                     row['Sugar_Tot_(g)'] = '0'
                  t['A'].append(float(row['Water_(g)']))
                  t['B'].append(float(row['Energ_Kcal']))
                  t['C'].append(float(row['Protein_(g)']))
                  t['D'].append(float(row['Lipid_Tot_(g)']))
                  t['E'].append(float(row['Carbohydrt_(g)']))
                  t['F'].append(float(row['Fiber_TD_(g)']))
                  t['G'].append(float(row['Sugar_Tot_(g)']))
          X = np.array([t['A'], t['B'], t['C'], t['D'], t['E'], t['F'], t['G']]).T
          y = [row['GI(Glucose= 100)'] for row in foodlist if not row['Water_(g)'] == '-1' and row['Similarity'] >= threshold]
          y = np.array(y).astype(np.float)

          X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, y, test_size=0.2,
                                                                                     random_state=5)
          clf = linear_model.LinearRegression()
          clf.fit(X_train, Y_train)
          print 'Estimated coefficient:', clf.coef_
          print 'Estimated intercept coefficient:', clf.intercept_
          global coef
          global intercept
          coef = clf.coef_
          intercept = clf.intercept_
          pred_train = clf.predict(X_train)
          pred_test = clf.predict(X_test)
          mseTrain = np.mean((Y_train - pred_train) ** 2)
          print 'Fit a model X_train, and calculate MSE with Y_train:', mseTrain
          mseTest = np.mean((Y_test - pred_test) ** 2)
          print 'Fit a model X_train, and calculate MSE with X_test, Y_test:', mseTest


          predicted = cross_val_predict(clf, X, y, cv=5)
          print '(Pearson correlation coefficient, 2-tailed p-value):', pearsonr(predicted, y)

          fig, ax = plt.subplots()
          ax.scatter(y, predicted)
          ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
          ax.set_xlabel('Measured')
          ax.set_ylabel('Predicted')
          ax.grid(True)

          global canvas
          canvas = FigureCanvasTkAgg(fig, master=root)
          plot_widget = canvas.get_tk_widget()
          plot_widget.grid(row=14, column=5, columnspan=3, rowspan=20)
          plot_widget.config(width=450, height=280)
    # switch the list box to select multiple state, make prediction of GI in a mixed meal
    else:
        global multiselected
        for index in multiselected:
            row = foodlist[int(index)]
            if row['Water_(g)'] == '':
               row['Water_(g)'] = '0'
            if row['Energ_Kcal'] == '':
               row['Energ_Kcal'] = '0'
            if row['Protein_(g)'] == '':
               row['Protein_(g)'] = '0'
            if row['Lipid_Tot_(g)'] == '':
               row['Lipid_Tot_(g)'] = '0'
            if row['Carbohydrt_(g)'] == '':
               row['Carbohydrt_(g)'] = '0'
            if row['Fiber_TD_(g)'] == '':
               row['Fiber_TD_(g)'] = '0'
            if row['Sugar_Tot_(g)'] == '':
               row['Sugar_Tot_(g)'] = '0'
            t['A'].append(float(row['Water_(g)']))
            t['B'].append(float(row['Energ_Kcal']))
            t['C'].append(float(row['Protein_(g)']))
            t['D'].append(float(row['Lipid_Tot_(g)']))
            t['E'].append(float(row['Carbohydrt_(g)']))
            t['F'].append(float(row['Fiber_TD_(g)']))
            t['G'].append(float(row['Sugar_Tot_(g)']))
            t['H'].append(row['Item'])
        if not 'coef' in globals():
            tkMessageBox.showwarning('Model Initialization', 'You haven\'t set a model.')
            switch = 0
        else:
            canvas.get_tk_widget().destroy()

            multifoods_prediction = intercept + sum(t['A'])*coef[0] + sum(t['B'])*coef[1] + sum(t['C'])*coef[2]\
                 + sum(t['D']) * coef[3] + sum(t['E']) * coef[4] + sum(t['F']) * coef[5] + sum(t['G']) * coef[6]
            prediction_display = '%.2f' % multifoods_prediction
            global predicted_GI
            predicted_GI = StringVar()
            Label(root, textvariable=predicted_GI).grid(row=12, column=5, sticky = W)
            predicted_GI.set('Predicited GI: '+ prediction_display)
            print 'Predicit GI for foods:', t['H']
    switch = 0
    listbox()


# widgets on the left side
Button(root, text=' LOAD  '.decode('gbk').encode('utf8'), font=('Arial', 16), command=load_csv).grid(row=0, column=0,
                                                                                                     sticky=W, padx=20,
                                                                                                     pady=10)
Button(root, text='  SET  '.decode('gbk').encode('utf8'), font=('Arial', 16), command=set).grid(row=9, column=7,
                                                                                                sticky=E)
set_notice = StringVar()
r = Label(root, textvariable=set_notice).grid(row=9, column=7, padx=20, sticky=W)
Button(root, text=' SAVE  '.decode('gbk').encode('utf8'), font=('Arial', 16), command=save_csv).grid(row=0, column=1,
                                                                                                     sticky=W, pady=10)
Button(root, text='  ADD    '.decode('gbk').encode('utf8'), font=('Arial', 16), command=add).grid(row=0, column=2,
                                                                                                 sticky=W, pady=10)
Button(root, text='DELETE'.decode('gbk').encode('utf8'), font=('Arial', 16), command=delete).grid(row=0, column=3,
                                                                                                  sticky=W, pady=10)

Button(root, text='MODEL'.decode('gbk').encode('utf8'), font=('Arial', 16), command=model).grid(row=11,
                                                                                                            column=7,
                                                                                                            sticky=E)
Button(root, text='SELECT MULTIPlE'.decode('gbk').encode('utf8'), font=('Arial', 16), command=select_multiple).grid(row=11,
                                                                                                            column=6)
USDA_foodname = StringVar()
Label(root, textvariable=USDA_foodname).grid(row=10, column=5, padx=20, pady=15, columnspan=3)
Label(root, text="Matching: ", font=("Arial", 16)).grid(row=10, column=5, sticky=W)


root.mainloop()
