from tkinter import Tk, Entry, Label, Button, filedialog, messagebox, Toplevel, OptionMenu, StringVar


class LoadGroups:

    def __init__(self, master):
        self.master = master
        master.title("Load classes")

        self.groupNames = []
        self.groupLabels = []
        self.groupButtons = []
        self.nameStruc = dict()
        self.fileList = dict()

        self.numClasses = 0
        self.instruction = Label(master, text="Enter number of groups in analysis (max 4): ")
        vcmd = master.register(self.validate)
        self.numClasses = Entry(master, validate="key", validatecommand=(vcmd, '%P'))
        self.goButton = Button(master, text="Go", command=lambda: self.update("go"))
        self.continueButton = Button(master, text="Continue", command=lambda: exit_routine(self))

        self.instruction.grid(row=0, column=0)
        self.numClasses.grid(row=0, column=1)
        self.goButton.grid(row=0, column=2)

    def validate(self, new_text):
        if not new_text:
            self.numClasses = 0
            return True
        try:
            self.numClasses = int(new_text)
            if 0 < self.numClasses < 5:
                return True
            else:
                return False
        except ValueError:
            return False

    def update(self, method):
        if method == "go":
            for name in self.groupNames:
                name.grid_forget()
            for label in self.groupLabels:
                label.grid_forget()
            for button in self.groupButtons:
                button.grid_forget()
            self.fileList.clear()

        self.continueButton.grid_forget()

        for k in range(self.numClasses):
            self.groupLabels.append(Label(self.master, text="Enter name of group " + str(k + 1) + ":"))
            self.groupNames.append(Entry(self.master))
            self.groupButtons.append(Button(self.master, text="Choose files",
                                            command=lambda k=k: load_group_files(self, k)))
            self.groupLabels[k].grid(row=k + 1, column=0)
            self.groupNames[k].grid(row=k + 1, column=1)
            self.groupButtons[k].grid(row=k + 1, column=2)

        self.continueButton.grid(row=6, column=1)


def exit_routine(obj):
    if len(obj.fileList) != len(obj.groupLabels):
        messagebox.showwarning("Warning", "Something does not add up. Are you sure?")
    else:
        obj.master.withdraw()
        defnames = Toplevel(obj.master)
        defnames.title("Define naming conventions")
        instructions = "We assume the naming convention is as follows: '**var1*var2*var3*.fext; where var1 is a group " \
                       "identifier, var2 is session identifier, var3 is sample/subject name with underscore between a " \
                       "short name for sample (e.g smpl) or subject (e.g. sub) etc. and the identifying number. " \
                       "Also fext is file extension." \
                       "and * stands for anything in between. Example s1_run1_ctr+hl_sub_23.txt \n"
        defnames.norms = StringVar(defnames)
        norm_choices = ["sqr", "tv"]
        defnames.trends = StringVar(defnames)
        trend_choices = ["none", "linear", "bias"]
        defnames.norms.set(norm_choices[0])
        defnames.trends.set(trend_choices[0])
        defnames.label_instr = Label(defnames, text=instructions, wraplength=700)
        defnames.label_sessions = Label(defnames, text="Enter names of sessions:")
        defnames.label_identifier = Label(defnames, text="Enter identifier (e.g sample, sub):")
        defnames.label_ext = Label(defnames, text="Enter file extension (e.g .txt, .csv):")
        defnames.label_runs = Label(defnames, text="Enter names of runs:")
        defnames.label_norms = Label(defnames, text="What norm?: ")
        defnames.label_trends = Label(defnames, text="What trend removal?:")
        defnames.identry = Entry(defnames)
        defnames.extentry = Entry(defnames)
        defnames.session_entry = Entry(defnames)
        defnames.run_entry = Entry(defnames)
        defnames.opt_norms = OptionMenu(defnames, defnames.norms, *norm_choices)
        defnames.opt_trends = OptionMenu(defnames, defnames.trends, *trend_choices)
        defnames.continue_button = Button(defnames, text="Continue", command=lambda: define_file_names(defnames, obj))
        defnames.label_instr.grid(row=0, column=0, columnspan=2)
        defnames.label_identifier.grid(row=1, column=0)
        defnames.label_ext.grid(row=2, column=0)
        defnames.identry.grid(row=1, column=1)
        defnames.extentry.grid(row=2, column=1)
        defnames.label_sessions.grid(row=3, column=0)
        defnames.label_runs.grid(row=4, column=0)
        defnames.session_entry.grid(row=3, column=1)
        defnames.run_entry.grid(row=4, column=1)
        defnames.label_norms.grid(row=5, column=0)
        defnames.label_trends.grid(row=6, column=0)
        defnames.opt_norms.grid(row=5, column=1)
        defnames.opt_trends.grid(row=6, column=1)
        defnames.continue_button.grid(row=7, column=0, columnspan=2)


def load_group_files(obj, k):
    if obj.groupNames[k].get() == '':
        messagebox.showerror("Error", "Group must have a name!")
    else:
        obj.fileList[obj.groupNames[k].get()] = filedialog.askopenfilenames(title="Select files for Group: " +
                                                                                  obj.groupNames[k].get())
        obj.groupLabels[k]['text'] = obj.groupLabels[k]['text'] + '[x]'


def define_file_names(childobj, parobj):
    sessions = childobj.session_entry.get().split(",")
    sessions = list(map(str.strip, sessions))
    if len(sessions) != len(set(sessions)):
        messagebox.showwarning("Duplicates", "The session names are not unique!")

    runs = childobj.run_entry.get().split(",")
    runs = list(map(str.strip, runs))
    if len(runs) != len(set(runs)):
        messagebox.showwarning("Duplicates", "The run names are not unique!")

    shortname = childobj.identry.get()
    fext = childobj.extentry.get()
    norm = childobj.norms.get()
    trend = childobj.trends.get()

    for group in parobj.fileList:
        for sample in parobj.fileList[group]:
            try:
                shortname + sample[(sample.index(shortname) + len(shortname)):sample.index(fext)]
            except ValueError:
                messagebox.showerror("Filename error", "Samples must be named in the format '*sub_###.txt': \n"
                                     + sample)
            try:
                next(run for run in runs if run in sample)
            except StopIteration:
                messagebox.showerror("Filename error", "Samples must be named in the format '*_run#_*.txt': \n"
                                     + sample)
                childobj.withdraw()
                parobj.master.quit()
                parobj.master.destroy()
                quit()

            try:
                next(session for session in sessions if session in sample)
            except StopIteration:
                messagebox.showwarning("Filename warning", "No session information found in filename: \n"
                                       + sample)

    parobj.nameStruc['runs'] = runs
    parobj.nameStruc['sessions'] = sessions
    parobj.nameStruc['short_name'] = shortname
    parobj.nameStruc['file_ext'] = fext
    parobj.nameStruc['norm'] = norm
    parobj.nameStruc['trend'] = trend
    parobj.nameStruc['groups'] = list(parobj.fileList.keys())

    childobj.withdraw()
    childobj.quit()
    parobj.master.quit()
    childobj.destroy()
    parobj.master.destroy()
