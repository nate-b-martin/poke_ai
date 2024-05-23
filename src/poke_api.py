import pokebase as pd
import random

class PokeAPI:
    def get_moves(self, pokemon):

        moves = pd.pokemon(pokemon).moves
        random_moves = random.sample(moves, 4)
        return [move.move.name for move in random_moves]
        # return [move.move.name for move in moves]



    def get_stats(self, pokemon):
        return {stat.stat.name: stat.base_stat for stat in pd.pokemon(pokemon).stats}


    def get_abilities(self, pokemon):
        abilities = pd.pokemon(pokemon).abilities
        ability_names = [ability.ability.name for ability in abilities]
        return ability_names
    
    def get_weaknesses(self, pokemon):
        weaknesses = pd.pokemon(pokemon).weaknesses
        weakness_types = [weakness.type.name for weakness in weaknesses]
        return weakness_types
    
    def get_resistances(self, pokemon):
        resistances = pd.pokemon(pokemon).resistances
        resistance_types = [resistance.type.name for resistance in resistances]
        return resistance_types

api = PokeAPI()
print(api.get_moves("charmander"))