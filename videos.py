"""
Created on Thu Nov 27 13:47:48 2025

@author: cloe
"""

import sys
import gurobipy as gp
from gurobipy import GRB


def lire_instance(input_path):
    """
    permet de :
    - lire une instance Hash Code 2017.
    - retourner
        V, E, R, C, X, video_sizes, endpoints, requests
    endpoints[e] = {"ld": latence_vers_dc, "caches": {cache_id: latence}}
    requests[r] = {"id": r, "video": v, "endpoint": e, "count": n}
    """
    print(f"Reading data from {input_path} ...")

    with open(input_path, "r", encoding="utf-8") as f:
        header = f.readline().split()
        if not header:
            raise ValueError("Empty input file")

        if len(header) != 5:
            raise ValueError("First line must contain 5 integers: V E R C X")

        V, E, R, C, X = map(int, header)

        # Tailles des videos
        sizes_line = f.readline().split()
        if len(sizes_line) != V:
            raise ValueError("Second line must contain V video sizes")
        video_sizes = list(map(int, sizes_line))

        # Endpoints 
        endpoints = []
        for e in range(E):
            ep_header = f.readline().split()
            if len(ep_header) != 2:
                raise ValueError("Endpoint header must contain 2 integers")
            ld = int(ep_header[0])
            k = int(ep_header[1])

            cache_connections = {}
            for _ in range(k):
                c_id, latency = map(int, f.readline().split())
                cache_connections[c_id] = latency

            endpoints.append({"ld": ld, "caches": cache_connections})

        # Requetes
        requests = []
        for r in range(R):
            line = f.readline().split()
            if len(line) != 3:
                raise ValueError("Request line must contain 3 integers")
            rv, re, rn = map(int, line)
            requests.append({"id": r, "video": rv, "endpoint": re, "count": rn})

    print(
        f"Data summary: V={V}, E={E}, R={R}, C={C}, X={X}"
    )
    return V, E, R, C, X, video_sizes, endpoints, requests


def build_model(V, E, R, C, X, video_sizes, endpoints, requests):
    """
    Construit le modele PLNE pour le probleme de video streaming.

    Variables:
      x[v,c] = 1 si la video v est stockee sur le cache c
      y[r,c] = 1 si la requete r est servie depuis le cache c

    Contraintes:
      - capacite pour chaque cache
      - chaque requete est servie par au plus un cache
      - y[r,c] <= x[v,c]
    """
    print("Creating Gurobi environment and model ")

    env = gp.Env()  # Environnement explicite comme dans le cours
    model = gp.Model("VideoStreaming", env=env)

    # Parametres comme dans l enonce du projet
    model.Params.LogToConsole = 1      # conserver le journal du solveur
    model.Params.MIPGap = 5e-3         # 0.5% de gap

    # Variables de decision x[v,c]
    x = {}  # (v,c) -> Var

    # Expressions de capacite par cache
    capacity = {c: gp.LinExpr() for c in range(C)}

    print("Creating decision variables and constraints ...")

    for request in requests:
        vid = request["video"]
        eid = request["endpoint"]
        num = request["count"]
        ld = endpoints[eid]["ld"]

        y_vars_for_request = []

        for cid, lc in endpoints[eid]["caches"].items():
            # Ne considerer que les caches plus rapides que le data center
            if lc >= ld:
                continue

            saving = (ld - lc) * num

            # Creer x[vid,cid] si elle n existe pas encore
            if (vid, cid) not in x:
                x[vid, cid] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"x_{vid}_{cid}",
                )
                capacity[cid].add(x[vid, cid], video_sizes[vid])

            # y[req_id,cid]: requete servie depuis ce cache
            y_rc = model.addVar(
                vtype=GRB.BINARY,
                obj=saving,  # coefficient dans la fonction objectif
                name=f"y_{request['id']}_{cid}",
            )
            y_vars_for_request.append(y_rc)

            # Contrainte de liaison: on ne peut servir depuis le cache
            # que si la video est stockee sur ce cache
            model.addConstr(
                y_rc <= x[vid, cid],
                name=f"link_{request['id']}_{cid}",
            )

        # Chaque requete peut etre servie par au plus un cache
        if y_vars_for_request:
            model.addConstr(
                gp.quicksum(y_vars_for_request) <= 1,
                name=f"assign_{request['id']}",
            )

    # Contraintes de capacite pour chaque cache
    print("Adding capacity constraints ...")
    for c in range(C):
        if capacity[c].size() > 0:
            model.addConstr(
                capacity[c] <= X,
                name=f"cap_cache_{c}",
            )

    # Sens de la fonction objectif: maximiser le gain de latence total
    model.ModelSense = GRB.MAXIMIZE

    model.update()

    print("Writing videos.mps ...")
    model.write("videos.mps")

    return model, env, x


def resol_restit(model, env, x, C, output_path="videos.out"):
    """
    permet de resoudre le modele et ecrit videos.out au format HashCode:
      premiere ligne: nombre de caches utilises
      puis: id cache video1 video2 ...
    """
    print("Optimizing model ...")
    model.optimize()

    if model.SolCount == 0:
        print("No solution found. videos.out will not be created.")
        # Liberation du modele et de l environnement
        model.dispose()
        env.dispose()
        return

    try:
        gap = model.MIPGap * 100.0
    except AttributeError:
        gap = None

    if gap is not None:
        print(f"Best objective: {model.ObjVal}, MIP gap: {gap:.4f}%")
    else:
        print(f"Best objective: {model.ObjVal}")

    # Extraire quelles videos sont stockees dans chaque cache
    cache_content = {c: [] for c in range(C)}
    for (vid, cid), var in x.items():
        if var.X > 0.5:
            cache_content[cid].append(vid)

    # Ne garder que les caches utilises et trier les id de video
    used_caches = {c: sorted(vlist) for c, vlist in cache_content.items() if vlist}

    print(f"Writing solution to {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write(f"{len(used_caches)}\n")
        for cid in sorted(used_caches.keys()):
            videos_str = " ".join(str(v) for v in used_caches[cid])
            f_out.write(f"{cid} {videos_str}\n")

    # Liberation des ressources Gurobi
    model.dispose()
    env.dispose()
    print("Done.")


def main(argv):
    if len(argv) != 2:
        print("Usage: python videos.py path_to_dataset.in")
        return 1

    input_path = argv[1]

    V, E, R, C, X, video_sizes, endpoints, requests = lire_instance(input_path)
    model, env, x = build_model(V, E, R, C, X, video_sizes, endpoints, requests)
    resol_restit(model, env, x, C, "videos.out")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
