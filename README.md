# Network Intrusion Detection Classifier
1. Install python 3.x
    * ``# apt-get install python3``
2. Install python requirements
    1. Install the python tkinter package for your operating system
        * ``# apt-get install python-tk``
    2. Install other required python packages
        * ``# pip install -r requirements.txt``
3. Run the IDS classifier
    1. Using classic machine learning techniques
        * ``python3 main.py --mode misuse --style classic``
        * ``python3 main.py --mode anomaly --style classic``
    2. Using deep learning techniques
        * ``python3 main.py --mode misuse --style neural``
        * ``python3 main.py --mode anomaly --style neural``
