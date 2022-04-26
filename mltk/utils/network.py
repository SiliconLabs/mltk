import socket
import random


def find_listening_port(port_range=None, host='localhost', socket_type='tcp', default_port=None):
    """Find an open listening port"""
    if port_range is None:
        port_range = (6000,65534)

    if socket_type == 'tcp':
        socket_protocol = socket.SOCK_STREAM
    elif socket_type == 'udp':
        socket_protocol = socket.SOCK_DGRAM
    else:
        raise Exception('Invalid socket_type argument, must be: tcp or udp')

    if default_port is not None:
        port = test_port(host, default_port, socket_protocol)
        if port != -1:
            return port

    searched_ports = [8080]
    for _ in range(100):
        port = random.randint(port_range[0], port_range[1])
        if port in searched_ports:
            continue 

        port = test_port(host, port, socket_protocol)
        if port != -1:
            return port

        searched_ports.append(port)

    raise Exception(f'Failed to find {socket_type} listening port for host={host}')



def test_port(host, port, socket_protocol):
    with socket.socket(socket.AF_INET, socket_protocol) as sock:
        try:
            sock.bind((host, port))
            return port
        except:
            pass 

    return -1
