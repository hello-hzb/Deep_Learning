objects = main.o read_file.o

main:$(objects)
	g++ -std=c++11 -o main $(objects)

main.o:main.cpp
	g++ -std=c++11 -c main.cpp -g

read_file.o:read_file.cpp
	g++ -std=c++11 -c read_file.cpp

clean:
	-rm -f *.o
