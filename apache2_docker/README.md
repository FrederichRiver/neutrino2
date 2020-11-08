# Docker config file

1. CMD to build the image.

In the director of docker project. Run this cmd.

```
sudo docker build -t <docker image:version> .
```

Note that there is a dote at the last, present it is a local directory.

2. CMD to create container.

```
sudo docker run -it --name=<name> -v <local_path:docker_path> -p <host:host_port:container_port>  <image:version> <start_point>
```
实例：
```
sudo docker run -it --name=apache_server -v /var/www/html:/var/www/html -p 127.0.0.1:8000:80 apache_serv:beta /bin/bash
```