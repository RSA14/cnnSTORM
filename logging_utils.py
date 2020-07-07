import logging

loglevel = logging.ERROR

def add(a, b):
    logging.basicConfig(filename='add.log',filemode = "w", level=loglevel)
    logging.info("Adding started")
    if not (a >= 0 and b >= 0):
        logging.error("Positives!")
        return("Error!")
    logging.info("adding completed")
    return (a + b)


if __name__ == "__main__":
    print(add(-2, 5))
