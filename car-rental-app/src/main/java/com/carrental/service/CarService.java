package com.carrental.service;

import com.carrental.dto.CarResponse;
import com.carrental.repository.CarRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class CarService {

    private final CarRepository carRepository;

    public List<CarResponse> getAvailableCars() {
        return carRepository.findByAvailableTrue().stream()
                .map(car -> CarResponse.builder()
                        .id(car.getId())
                        .brand(car.getBrand())
                        .model(car.getModel())
                        .year(car.getYear())
                        .pricePerDay(car.getPricePerDay())
                        .available(car.getAvailable())
                        .build())
                .toList();
    }

    public List<CarResponse> getAllCars() {
        return carRepository.findAll().stream()
                .map(car -> CarResponse.builder()
                        .id(car.getId())
                        .brand(car.getBrand())
                        .model(car.getModel())
                        .year(car.getYear())
                        .pricePerDay(car.getPricePerDay())
                        .available(car.getAvailable())
                        .build())
                .toList();
    }
}
