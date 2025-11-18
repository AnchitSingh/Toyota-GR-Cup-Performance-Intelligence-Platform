# config.py - Complete track configurations

from pathlib import Path

TRACKS = {
    'Sebring': {
        'path': Path('sebring/Sebring'),
        'races': ['R1', 'R2'],
        'has_race_dirs': True,
        'expected_corners': (12, 30),
        'typical_lap_time': (120, 160)
    },
    'Sonoma': {
        'path': Path('sonoma/Sonoma'),
        'races': ['R1', 'R2'],
        'has_race_dirs': True,
        'expected_corners': (8, 25),
        'typical_lap_time': (115, 150)
    },
    'Barber': {
        'path': Path('barber-motorsports-park/barber'),
        'races': ['R1', 'R2'],
        'has_race_dirs': False,  # Files directly in barber/
        'expected_corners': (10, 30),
        'typical_lap_time': (110, 145)
    },
    'Indianapolis': {
        'path': Path('indianapolis/indianapolis'),
        'races': ['R1', 'R2'],
        'has_race_dirs': False,  # Files directly in indianapolis/
        'expected_corners': (8, 25),
        'typical_lap_time': (115, 155)
    },
    'VIR': {
        'path': Path('virginia-international-raceway/VIR'),
        'races': ['R1', 'R2'],
        'has_race_dirs': True,
        'expected_corners': (6, 20),
        'typical_lap_time': (115, 145)
    },
    'Road America': {
        'path': Path('road-america/Road America'),
        'races': ['R1', 'R2'],
        'has_race_dirs': True,
        'expected_corners': (8, 30),
        'typical_lap_time': (135, 175)
    },
    'COTA': {
        'path': Path('circuit-of-the-americas/COTA'),
        'races': ['R1', 'R2'],
        'has_race_dirs': True,
        'expected_corners': (10, 35),
        'typical_lap_time': (130, 165)
    }
}
