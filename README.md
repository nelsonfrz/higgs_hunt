# Higgs-Suche in CERN Open Data
Das Ziel dieses Projekts ist der Nachweis des Higgs-Bosons anhand von CMS-Detektor-Datensätzen, die über CERN Open Data verfügbar sind. Der Fokus liegt auf dem H→ZZ→4l Zerfallskanal, wobei speziell Elektronen und Myonen als Endprodukte analysiert werden. Durch eine eigenentwickelte Software werden Datensätze heruntergeladen und Kombinationen von 4 Leptonen selektiert, um den Zerfallskanal zu rekonstruieren. Anschließend werden die Kombinationen auf charakteristische Eigenschaften des Higgs-Bosons, wie die invariante Masse, Energie und Isolation der Zerfallsprodukte, mittels einer eigens entwickelten interaktiven Software in Echtzeit gefiltert, die sich in einem virtuellen Docker-Container durch ein eigenes Dockerfile erzeugen lässt. Dabei werden die Datenanalyse Werkzeuge Uproot, Pandas, NumPy sowie Matplotlib benutzt. Die Ergebnisse einer Analyse mit 100 Millionen Ereignisse zeigen ein deutliches Signal bei einer Masse von etwa 125 GeV, was die Existenz des Higgs-Bosons nachweist.

# Dokumentation
## Getting Started
Um die Analyse auszuführen wird eine Installation von [Docker](https://www.docker.com/) vorausgesetzt.

Folgender Befehl muss in der Kommandozeile im Stammverzeichnis `higgs_hunt/` ausgeführt werden, um ein Docker-Image mit der Analyse zu bauen:
```
docker build -t higgshunt .
```

Nun muss ein Docker-Container vom Docker-Image erzeugt werden mit folgendem Befehl in der Kommandozeile:
```
docker run -p 8888:8888 -e NOTEBOOK_ARGS="--ip='*' --LabApp.token='' --LabApp.password=''" -d higgshunt
```

Nun sollte in einem beliebigen Web-Browser unter http://127.0.0.1:8888/ eine Jupyter Lab Entwicklungsumgebgung erreichbar sein mit der Analyse.