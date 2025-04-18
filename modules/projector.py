from tkinter import *
from tkinter import ttk

# a subclass of Canvas for dealing with resizing of windows
class ResizingCanvas(Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.bind("<Configure>", self.on_resize)
        self.width = self.winfo_reqwidth()
        self.height = self.winfo_reqheight()

    def on_resize(self, event):
        # determine the ratio of old width/height to new width/height
        wscale = float(event.width) / self.width
        hscale = float(event.height) / self.height
        self.width, self.height = event.width, event.height
        # resize the canvas 
        self.config(width=self.width, height=self.height)
        # rescale all the objects tagged with the "all" tag
        self.scale("all", 0, 0, wscale, hscale)

class Projector:
    def __init__(self, size_window=(600, 400)):
        self.root = Tk()
        self.root.title("Bincase - Demo - projector")
        myframe = Frame(self.root)
        myframe.pack(fill=BOTH, expand=YES)
        
        # Get the current screen width and height
        self.screen_width, self.screen_height = size_window
        
        self.canvas = ResizingCanvas(
            myframe,
            bd=-2,
            width=self.screen_width,
            height=self.screen_height,
            bg="black"
        )
        self.canvas.pack(fill=BOTH, expand=YES)