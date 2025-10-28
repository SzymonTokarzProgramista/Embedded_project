# ------------------- Build Stage -------------------
FROM python:3.11-slim AS build

ENV DEBIAN_FRONTEND=noninteractive

# System deps do kompilacji + USB + GUI-dev potrzebne do builda librealsense
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential libssl-dev libusb-1.0-0-dev pkg-config \
    libglfw3-dev libgtk-3-dev libgl1-mesa-dev libglu1-mesa-dev udev \
 && rm -rf /var/lib/apt/lists/*

# Venv (tu wyląduje pyrealsense2 + reszta)
ENV VENV=/venv
ENV PATH="$VENV/bin:$PATH"
RUN python -m venv $VENV

# Pythonowe zależności projektu (bez pyrealsense2 – będzie zbudowane)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# Zbuduj librealsense + zainstaluj binding Pythona do /venv
WORKDIR /opt
RUN git clone --depth 1 https://github.com/IntelRealSense/librealsense.git
WORKDIR /opt/librealsense/build

ENV PY_SITE=/venv/lib/python3.11/site-packages
RUN cmake .. \
    -DFORCE_RSUSB_BACKEND=ON \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DPYTHON_EXECUTABLE=$VENV/bin/python \
    -DPYTHON_INSTALL_DIR=${PY_SITE} \
    -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF \
 && make -j$(nproc) && make install && ldconfig

# Udev rules (opcjonalne w kontenerze, ale nie przeszkadzają)
RUN cp /opt/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/ || true


# ------------------- Final Stage -------------------
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# ❗ RUNTIME dla OpenCV i RealSense (to właśnie brakowało: libGL itd.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libusb-1.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Skopiuj venv + /usr/local (librealsense *.so) + udev rules
ENV VENV=/venv
ENV PATH="$VENV/bin:$PATH"
COPY --from=build $VENV $VENV
COPY --from=build /usr/local /usr/local
COPY --from=build /etc/udev/rules.d/99-realsense-libusb.rules /etc/udev/rules.d/99-realsense-libusb.rules

# Aplikacja
WORKDIR /Embedded_project
COPY . .

# PYTHONPATH, by działały absolutne importy: `from vision_system...` i `from api...`
ENV PYTHONPATH=/Embedded_project/src

EXPOSE 5000
CMD ["python", "-u", "app.py"]
