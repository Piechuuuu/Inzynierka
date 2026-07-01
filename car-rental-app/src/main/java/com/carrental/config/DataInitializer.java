package com.carrental.config;

import com.carrental.entity.AppUser;
import com.carrental.entity.Car;
import com.carrental.enums.Role;
import com.carrental.repository.AppUserRepository;
import com.carrental.repository.CarRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.boot.CommandLineRunner;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.util.List;

@Component
@RequiredArgsConstructor
public class DataInitializer implements CommandLineRunner {

    private final AppUserRepository userRepository;
    private final CarRepository carRepository;
    private final PasswordEncoder passwordEncoder;

    @Override
    public void run(String... args) {
        initUsers();
        initCars();
    }

    private void initUsers() {
        List.of(
                AppUser.builder()
                        .username("admin")
                        .password(passwordEncoder.encode("admin123"))
                        .role(Role.ROLE_ADMIN)
                        .age(35)
                        .build(),
                AppUser.builder()
                        .username("jan")
                        .password(passwordEncoder.encode("jan123"))
                        .role(Role.ROLE_USER)
                        .age(30)
                        .build(),
                AppUser.builder()
                        .username("anna")
                        .password(passwordEncoder.encode("anna123"))
                        .role(Role.ROLE_USER)
                        .age(22)
                        .build()
        ).forEach(userRepository::save);
    }

    private void initCars() {
        List.of(
                Car.builder().brand("Toyota").model("Corolla").year(2022).pricePerDay(BigDecimal.valueOf(150)).available(true).build(),
                Car.builder().brand("BMW").model("320i").year(2023).pricePerDay(BigDecimal.valueOf(300)).available(true).build(),
                Car.builder().brand("Volkswagen").model("Golf").year(2021).pricePerDay(BigDecimal.valueOf(200)).available(true).build(),
                Car.builder().brand("Audi").model("A4").year(2023).pricePerDay(BigDecimal.valueOf(350)).available(true).build(),
                Car.builder().brand("Ford").model("Focus").year(2020).pricePerDay(BigDecimal.valueOf(120)).available(false).build()
        ).forEach(carRepository::save);
    }
}
