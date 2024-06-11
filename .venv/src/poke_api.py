import pokebase as pd
import random

class PokeAPI:
    def get_moves(self, pokemon):

        moves = pd.pokemon(pokemon.lower()).moves
        random_moves = random.sample(moves, 4)
        return [move.move.name for move in random_moves]
        # return [move.move.name for move in moves]

    def get_move_data(self, move_name, attribute):
        move = pd.move(move_name.lower())
        return {attribute: move[attribute]}



    def get_stats(self, pokemon):
        return {stat.stat.name: stat.base_stat for stat in pd.pokemon(pokemon.lower()).stats}


    def get_abilities(self, pokemon):
        abilities = pd.pokemon(pokemon.lower()).abilities
        ability_names = [ability.ability.name for ability in abilities]
        return ability_names
    
    def get_weaknesses(self, pokemon):
        weaknesses = pd.pokemon(pokemon.lower()).weaknesses
        weakness_types = [weakness.type.name for weakness in weaknesses]
        return weakness_types
    
    def get_resistances(self, pokemon):
        resistances = pd.pokemon(pokemon.lower()).resistances
        resistance_types = [resistance.type.name for resistance in resistances]
        return resistance_types

    def get_pokemon(self, pokemon):
        current_pokemon = pd.pokemon(pokemon.lower())
        return {
            "name": current_pokemon.name,
            "id": current_pokemon.id,
            "height": current_pokemon.height,
            "weight": current_pokemon.weight,
            "base_experience": current_pokemon.base_experience,
            "stats": self.get_stats(current_pokemon.name),
            "abilities": self.get_abilities(current_pokemon.name),
            "moves": self.get_moves(current_pokemon.name)
        }

if __name__ == "__main__":
    api = PokeAPI()
    pikachu = api.get_pokemon("pikachu")
    print(pikachu)
    
