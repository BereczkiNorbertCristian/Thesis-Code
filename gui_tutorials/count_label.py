
import tkinter as tk

counter = 0
def counter_label(label):
    def count():
        print("Clicked")
        global counter
        counter += 1
        label.config(text=str(counter))
    return count

root = tk.Tk()
root.title("Counting Seconds")
label = tk.Label(root,fg="green")
label.pack()
callback = counter_label(label)
callback()
button2 = tk.Button(root,text='Increment',width=25,command=callback)
button2.pack()
button = tk.Button(root,text='Exit',width=25,command=root.destroy)
button.pack()
root.mainloop()
