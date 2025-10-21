# ------------------- Build Stage -------------------
FROM python:3.11-slim AS build

# System deps do kompilacji + USB + minimalne GUI liby wymagane przez librealsense
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential libssl-dev libusb-1.0-0-dev pkg-config \
    libglfw3-dev libgtk-3-dev libgl1-mesa-dev libglu1-mesa-dev udev \
    && rm -rf /var/lib/apt/lists/*

# Venv dla aplikacji (i tu wlejemy pyrealsense2)
ENV VENV=/venv
ENV PATH="$VENV/bin:$PATH"
RUN python -m venv $VENV

# Pythonowe zależności projektu (UWAGA: bez pyrealsense2)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Zbuduj librealsense + zainstaluj binding Pythona prosto do /venv
WORKDIR /opt
RUN git clone --depth 1 https://github.com/IntelRealSense/librealsense.git
WORKDIR /opt/librealsense/build

# Ścieżka do site-packages w naszym venv
ENV PY_SITE=/venv/lib/python3.11/site-packages
RUN cmake .. \
    -DFORCE_RSUSB_BACKEND=ON \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DPYTHON_EXECUTABLE=$VENV/bin/python \
    -DPYTHON_INSTALL_DIR=${PY_SITE} \
    -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF && \
    make -j4 && make install && ldconfig

# (opcjonalnie) reguły udev w kontenerze
RUN cp /opt/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/ || true


# ------------------- Final Stage -------------------
FROM python:3.11-slim

# Skopiuj venv z zainstalowanym pyrealsense2 i biblioteki z /usr/local
ENV VENV=/venv
ENV PATH="$VENV/bin:$PATH"
COPY --from=build $VENV $VENV
COPY --from=build /usr/local /usr/local
COPY --from=build /etc/udev/rules.d/99-realsense-libusb.rules /etc/udev/rules.d/99-realsense-libusb.rules

# Aplikacja
WORKDIR /Embedded_project
COPY . .

EXPOSE 5000
CMD ["python", "-u", "app.py"]
