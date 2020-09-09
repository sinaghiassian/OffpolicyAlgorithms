import pyglet

window = pyglet.window.Window(800, 600)


episode_number_label = pyglet.text.Label('Hello, world',
                          font_name='Times New Roman',
                          font_size=36,
                          x=window.width // 2, y=window.height // 2,
                          anchor_x='center', anchor_y='center')
@window.event
def on_draw():
    window.clear()
    #label.draw()
    player_ship.draw()



from pyglet.gl import GLubyte

pixels = [
    255, 0, 0,      0, 255, 0,      0, 0, 255,     # RGB values range from
    255, 0, 0,      255, 0, 0,      255, 0, 0,     # 0 to 255 for each color
    255, 0, 0,      255, 0, 0,      255, 0, 0,     # component.
]
rawData = (GLubyte * len(pixels))(*pixels)
imageData = pyglet.image.ImageData(3, 3, 'RGB', rawData)
player_ship = pyglet.sprite.Sprite(img=imageData, x=400, y=300)

if __name__ == "__main__":
    pyglet.app.run()

