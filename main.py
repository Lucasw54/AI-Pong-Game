import numpy as np
import tensorflow as tf
import pygame
from sklearn.model_selection import train_test_split

# Initialize Pygame
pygame.init()

# Set the window size
window_size = (640, 480)

# Set the number of games to simulate
num_games = 10000

# Initialize arrays to store the data and labels
X = np.zeros((num_games, 3))
Y = np.zeros((num_games, 1))

# Loop through the number of games to simulate
for i in range(num_games):
  # Initialize the game state
  ball_x = 0.5
  ball_y = 0.5
  ball_vx = 0.03
  ball_vy = 0.01
  paddle_y = 0.5
  paddle_v = 0.04
  
  # Create the Pygame window
  screen = pygame.display.set_mode(window_size)
  
  # Loop through the time steps in the game
  for t in range(50):
    # Update the ball position
    ball_x += ball_vx
    ball_y += ball_vy
    
    # Check for collision with the walls
    if ball_y < 0 or ball_y > 1:
      ball_vy *= -1
    
    # Update the paddle position
    paddle_y += paddle_v
    
    # Check for collision with the walls
    if paddle_y < 0:
      paddle_y = 0
    elif paddle_y > 1:
      paddle_y = 1
      
    # Check for collision with the paddle
    if ball_x < 0 and ball_y > paddle_y and ball_y < paddle_y + 0.2:
      ball_vx *= -1
    
    # Store the game state as a feature
    X[i, 0] = ball_x
    X[i, 1] = ball_y
    X[i, 2] = paddle_y
    
    # Store the action as a label
    if paddle_v > 0:
      Y[i, 0] = 1
    else:
      Y[i, 0] = 0
    
    # Draw the ball and paddle on the screen
    ball_x_px = int(ball_x * window_size[0])
    ball_y_px = int(ball_y * window_size[1])
    paddle_y_px = int(paddle_y * window_size[1])
    pygame.draw.circle(screen, (255, 255, 255), (ball_x_px, ball_y_px), 5)
    pygame.draw.rect(screen, (255, 255, 255), (0, paddle_y_px, 10, 20))
    
    # Update the screen
    pygame.display.flip()

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)

# Save the model
model.save('pong_model.h5')
