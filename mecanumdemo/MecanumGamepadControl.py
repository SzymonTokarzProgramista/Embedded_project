import pygame
import math
import mecanum_pb2
import serial
import time
import os

os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Użyj "dummy" drivera do testów bez wyświetlacza

pygame.init()
pygame.joystick.init()

# Ustawienia okna
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Gamepad Visualization")
clock = pygame.time.Clock()

# Kolory
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)

# wykrycie liczby podłączonych padów
joystick_count = pygame.joystick.get_count()
print("Liczba padów:", joystick_count)

joystick = None
if joystick_count > 0:
    joystick = pygame.joystick.Joystick(0)  # pierwszy pad
    joystick.init()
    print("Wykryto pad:", joystick.get_name())

# Zmienne do przechowywania stanu padów
left_stick_x = 0.0
left_stick_y = 0.0
right_stick_x = 0.0
right_stick_y = 0.0
button_states = {}

# Ustawienia komunikacji z robotem
try:
    ser = serial.Serial('/dev/ttyUSB0', 19200, timeout=1)  # Dostosuj port według potrzeb
    time.sleep(2)
    print("Połączono z robotem przez port /dev/ttyUSB0")
    robot_connected = True
except serial.SerialException:
    print("Nie można połączyć z robotem - sprawdź połączenie USB")
    robot_connected = False
    ser = None

# Parametry sterowania robotem
MAX_SPEED = 500  # Maksymalna prędkość w mm/s
MAX_OMEGA = 2.0  # Maksymalna prędkość obrotu w rad/s
last_command_time = 0
COMMAND_INTERVAL = 0.1  # Wysyłaj komendy co 100ms

def send_robot_command(speed_mmps, direction_rad, omega_rads):
    """Wysyła komendę do robota przez port szeregowy"""
    global robot_connected, ser
    
    if not robot_connected or ser is None:
        return
    
    try:
        request = mecanum_pb2.ControlRequest()
        request.speed_mmps = int(speed_mmps)
        request.rad = float(direction_rad)
        request.omega = float(omega_rads)
        
        # Serializacja wiadomości
        serialized_data = request.SerializeToString()
        
        # Wysłanie do Arduino
        ser.write(serialized_data)
    except Exception as e:
        print(f"Błąd podczas wysyłania komendy: {e}")

def gamepad_to_robot_control(left_x, left_y, right_x):
    """Konwertuje wejścia z gamepada na parametry sterowania robotem"""
    # Lewy drążek kontroluje ruch (speed i direction)
    # Prawy drążek X kontroluje obrót (omega)
    
    # Oblicz prędkość na podstawie amplitudy lewego drążka
    speed = math.sqrt(left_x**2 + left_y**2) * MAX_SPEED
    
    # Oblicz kąt kierunku (0 = prawo, pi/2 = przód, pi = lewo, 3*pi/2 = tył)
    if speed > 10:  # Minimalna prędkość aby uniknąć drgań
        direction = math.atan2(-left_y, left_x)  # Odwróć Y dla właściwego kierunku
        if direction < 0:
            direction += 2 * math.pi
    else:
        direction = 0
        speed = 0
    
    # Prędkość obrotu z prawego drążka
    omega = -right_x * MAX_OMEGA  # Odwróć znak dla intuicyjnego sterowania
    
    return speed, direction, omega

def draw_analog_stick(surface, center_x, center_y, x_value, y_value, label):
    """Rysuje wizualizację drążka analogowego"""
    # Zewnętrzny okrąg (granice)
    pygame.draw.circle(surface, GRAY, (center_x, center_y), 80, 3)
    
    # Pozycja drążka (skalowana do rozmiaru okręgu)
    stick_x = center_x + int(x_value * 70)
    stick_y = center_y + int(y_value * 70)
    
    # Linia od środka do pozycji drążka
    pygame.draw.line(surface, BLUE, (center_x, center_y), (stick_x, stick_y), 2)
    
    # Kropka pokazująca pozycję drążka
    pygame.draw.circle(surface, RED, (stick_x, stick_y), 8)
    
    # Środkowa kropka
    pygame.draw.circle(surface, GREEN, (center_x, center_y), 5)
    
    # Etykieta
    font = pygame.font.Font(None, 24)
    text = font.render(label, True, WHITE)
    surface.blit(text, (center_x - 40, center_y - 110))
    
    # Wartości liczbowe
    value_text = font.render(f"X: {x_value:.2f}", True, WHITE)
    surface.blit(value_text, (center_x - 50, center_y + 100))
    value_text = font.render(f"Y: {y_value:.2f}", True, WHITE)
    surface.blit(value_text, (center_x - 50, center_y + 120))

def draw_button(surface, x, y, width, height, is_pressed, label):
    """Rysuje wizualizację przycisku"""
    color = GREEN if is_pressed else GRAY
    pygame.draw.rect(surface, color, (x, y, width, height))
    pygame.draw.rect(surface, WHITE, (x, y, width, height), 2)
    
    # Etykieta przycisku
    font = pygame.font.Font(None, 20)
    text = font.render(label, True, WHITE)
    text_rect = text.get_rect(center=(x + width//2, y + height//2))
    surface.blit(text, text_rect)

running = True
while running:
    current_time = time.time()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # przyciski
        if event.type == pygame.JOYBUTTONDOWN:
            print("Wciśnięto przycisk:", event.button)
            button_states[event.button] = True
        if event.type == pygame.JOYBUTTONUP:
            print("Zwolniono przycisk:", event.button)
            button_states[event.button] = False

        # osie analogów
        if event.type == pygame.JOYAXISMOTION:
            if event.axis == 0:  # Lewy drążek X
                left_stick_x = event.value
            elif event.axis == 1:  # Lewy drążek Y
                left_stick_y = event.value
            elif event.axis == 2:  # Prawy drążek X (może być inna oś w zależności od pada)
                right_stick_x = event.value
            elif event.axis == 3:  # Prawy drążek Y
                right_stick_y = event.value

        # krzyżak (hat)
        if event.type == pygame.JOYHATMOTION:
            print("Hat:", event.value)
    
    # Wysyłanie komend do robota co określony interwał
    if current_time - last_command_time >= COMMAND_INTERVAL and joystick:
        speed, direction, omega = gamepad_to_robot_control(left_stick_x, left_stick_y, right_stick_x)
        send_robot_command(speed, direction, omega)
        last_command_time = current_time

    # Czyszczenie ekranu
    screen.fill(BLACK)
    
    # Rysowanie wizualizacji
    if joystick:
        # Lewy drążek analogowy
        draw_analog_stick(screen, 200, 200, left_stick_x, left_stick_y, "Lewy drążek")
        
        # Prawy drążek analogowy
        draw_analog_stick(screen, 600, 200, right_stick_x, right_stick_y, "Prawy drążek")
        
        # Przyciski
        button_y = 400
        for i in range(min(12, joystick.get_numbuttons())):  # Maksymalnie 12 przycisków
            is_pressed = button_states.get(i, False)
            button_x = 50 + (i * 60)
            draw_button(screen, button_x, button_y, 50, 30, is_pressed, str(i))
        
        # Informacje o padzie
        font = pygame.font.Font(None, 36)
        title_text = font.render("Gamepad Visualization", True, WHITE)
        screen.blit(title_text, (WINDOW_WIDTH//2 - 150, 50))
        
        info_font = pygame.font.Font(None, 24)
        info_text = info_font.render(f"Pad: {joystick.get_name()}", True, WHITE)
        screen.blit(info_text, (50, 500))
        
        axes_text = info_font.render(f"Osie: {joystick.get_numaxes()}", True, WHITE)
        screen.blit(axes_text, (50, 520))
        
        buttons_text = info_font.render(f"Przyciski: {joystick.get_numbuttons()}", True, WHITE)
        screen.blit(buttons_text, (50, 540))
        
        # Status robota i parametry sterowania
        robot_status_text = info_font.render(f"Robot: {'Połączony' if robot_connected else 'Rozłączony'}", True, GREEN if robot_connected else RED)
        screen.blit(robot_status_text, (400, 500))
        
        if robot_connected:
            speed, direction, omega = gamepad_to_robot_control(left_stick_x, left_stick_y, right_stick_x)
            speed_text = info_font.render(f"Prędkość: {speed:.0f} mm/s", True, WHITE)
            screen.blit(speed_text, (400, 520))
            
            direction_deg = math.degrees(direction)
            direction_text = info_font.render(f"Kierunek: {direction_deg:.0f}°", True, WHITE)
            screen.blit(direction_text, (400, 540))
            
            omega_text = info_font.render(f"Obrót: {omega:.2f} rad/s", True, WHITE)
            screen.blit(omega_text, (400, 560))
    else:
        # Komunikat gdy brak pada
        font = pygame.font.Font(None, 48)
        no_gamepad_text = font.render("Brak podłączonego gamepada", True, WHITE)
        text_rect = no_gamepad_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2))
        screen.blit(no_gamepad_text, text_rect)

    # Odświeżenie ekranu
    pygame.display.flip()
    clock.tick(60)  # 60 FPS

# Zatrzymanie robota przed zamknięciem
if robot_connected and ser:
    try:
        # Wyślij komendę stop (prędkość = 0)
        send_robot_command(0, 0, 0)
        ser.close()
        print("Połączenie z robotem zamknięte")
    except Exception as e:
        print(f"Błąd podczas zamykania połączenia: {e}")

pygame.quit()
