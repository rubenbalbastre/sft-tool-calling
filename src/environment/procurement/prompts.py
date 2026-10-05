"""Multilingual surface realizations for structured procurement scenarios."""

from copy import deepcopy


LANGUAGES = ("English", "Spanish", "German", "French")
TEMPLATES_PER_LANGUAGE = 10


ROUTES = {
    "English": {
        "direct_supplier": "Evaluate only supplier {requested_supplier_id}.",
        "open_search": "Search for the best supplier option.",
        "compliance_first": "Use only suppliers certified to ISO-14001.",
        "preferred_with_fallback": "Request an offer from {preferred_supplier_id} first, then use another supplier if it cannot satisfy every requirement.",
        "no_feasible_option": "If all available options violate the requirements, report that no feasible option exists.",
    },
    "Spanish": {
        "direct_supplier": "Evalúa únicamente al proveedor {requested_supplier_id}.",
        "open_search": "Busca la mejor opción entre los proveedores disponibles.",
        "compliance_first": "Utiliza solo proveedores con certificación ISO-14001.",
        "preferred_with_fallback": "Solicita primero una oferta a {preferred_supplier_id} y recurre a otro proveedor si no cumple todos los requisitos.",
        "no_feasible_option": "Si todas las opciones disponibles incumplen los requisitos, indica que no existe ninguna opción viable.",
    },
    "German": {
        "direct_supplier": "Prüfe ausschließlich den Lieferanten {requested_supplier_id}.",
        "open_search": "Suche die beste Option unter den verfügbaren Lieferanten.",
        "compliance_first": "Berücksichtige nur Lieferanten mit ISO-14001-Zertifizierung.",
        "preferred_with_fallback": "Fordere zuerst ein Angebot von {preferred_supplier_id} an und weiche auf einen anderen Lieferanten aus, falls nicht alle Anforderungen erfüllt werden.",
        "no_feasible_option": "Falls alle verfügbaren Optionen die Anforderungen verletzen, melde, dass keine geeignete Option existiert.",
    },
    "French": {
        "direct_supplier": "Évalue uniquement le fournisseur {requested_supplier_id}.",
        "open_search": "Recherche la meilleure option parmi les fournisseurs disponibles.",
        "compliance_first": "Utilise uniquement des fournisseurs certifiés ISO-14001.",
        "preferred_with_fallback": "Demande d'abord une offre à {preferred_supplier_id}, puis choisis un autre fournisseur s'il ne respecte pas toutes les contraintes.",
        "no_feasible_option": "Si toutes les options disponibles enfreignent les contraintes, indique qu'aucune option viable n'existe.",
    },
}


PROMPT_TEMPLATES = {
    "English": (
        "We need {quantity} {unit} of {material_id} delivered to {destination} by {required_date}. Total cost must not exceed EUR {budget}, and delivery reliability must be at least {reliability}. {route} Prefer lower carbon emissions, then lower cost. Use the tools and finish with a submitted plan or a no-feasible-option report.",
        "Please arrange {quantity} {unit} of {material_id} for {destination}, arriving no later than {required_date}. Keep the complete order below EUR {budget} with delivery reliability of {reliability} or higher. {route} Of the feasible choices, prioritize carbon impact before price. Complete the task through the tools.",
        "Can you source {material_id} for us? The requirement is {quantity} {unit} in {destination} by {required_date}, with a maximum total spend of EUR {budget} and at least {reliability} delivery reliability. {route} Minimize emissions first and cost second, then submit the result using the tools.",
        "Procurement request: {material_id}, {quantity} {unit}, destination {destination}, deadline {required_date}. Hard limits are EUR {budget} total and {reliability} minimum delivery reliability. {route} Among valid options, favor lower carbon and then lower cost. Resolve the request with the available tools.",
        "For delivery to {destination} by {required_date}, obtain {quantity} {unit} of {material_id}. Do not exceed EUR {budget}; require delivery reliability of at least {reliability}. {route} Carbon reduction is the primary preference and price is secondary. Finish with the appropriate terminal tool.",
        "I need help placing an order for {quantity} {unit} of {material_id}. It has to reach {destination} on or before {required_date}, cost no more than EUR {budget} in total, and achieve at least {reliability} delivery reliability. {route} Prefer the greener feasible choice, using price as the next criterion. Handle the request with the available tools.",
        "Find a compliant way to supply {destination} with {quantity} {unit} of {material_id} by {required_date}. The all-in budget is EUR {budget} and the delivery reliability floor is {reliability}. {route} Once the mandatory conditions are met, optimize for carbon and then cost. Return the final decision through a terminal tool.",
        "We are planning a shipment of {material_id}: {quantity} {unit} are required in {destination} by {required_date}. Reject anything above EUR {budget} total or below {reliability} delivery reliability. {route} For the remaining choices, lower emissions matter more than lower price. Use the tools to complete the selection.",
        "Source {quantity} {unit} of {material_id} for {destination}. Deadline: {required_date}; total budget ceiling: EUR {budget}; minimum delivery reliability: {reliability}. {route} Choose by lowest carbon first and lowest cost second, subject to those limits, and submit the outcome with the tools.",
        "Could you compare the available procurement options for {quantity} {unit} of {material_id}, needed at {destination} by {required_date}? We cannot spend over EUR {budget} overall or accept delivery reliability below {reliability}. {route} Sustainability is the main preference, followed by cost. Finish the workflow using the tools.",
    ),
    "Spanish": (
        "Necesitamos {quantity} {unit} de {material_id} entregados en {destination} antes del {required_date}. El coste total no puede superar EUR {budget} y la fiabilidad de la entrega debe ser como mínimo del {reliability}. {route} Prioriza menos emisiones de carbono y después un menor coste. Usa las herramientas y finaliza con un plan o indicando que no hay una opción viable.",
        "Gestiona la compra de {quantity} {unit} de {material_id} para {destination}, con llegada máxima el {required_date}. Mantén el pedido completo por debajo de EUR {budget} y exige al menos un {reliability} de fiabilidad en la entrega. {route} Entre las opciones viables, valora primero el impacto de carbono y luego el precio. Completa la tarea mediante las herramientas.",
        "¿Puedes conseguir {material_id}? Hacen falta {quantity} {unit} en {destination} para el {required_date}, con un gasto total máximo de EUR {budget} y una fiabilidad de entrega mínima del {reliability}. {route} Minimiza primero las emisiones y después el coste, y presenta el resultado usando las herramientas.",
        "Solicitud de compras: {material_id}, {quantity} {unit}, destino {destination}, fecha límite {required_date}. Los límites obligatorios son EUR {budget} en total y un {reliability} de fiabilidad mínima de entrega. {route} Entre las alternativas válidas, favorece menos carbono y luego menor coste. Resuelve la solicitud con las herramientas disponibles.",
        "Para entregar en {destination} antes del {required_date}, adquiere {quantity} {unit} de {material_id}. No superes EUR {budget} y exige una fiabilidad de entrega de al menos {reliability}. {route} La reducción de carbono es prioritaria y el precio es secundario. Termina con la herramienta final adecuada.",
        "Necesito ayuda para tramitar un pedido de {quantity} {unit} de {material_id}. Debe llegar a {destination} como máximo el {required_date}, costar en total menos de EUR {budget} y ofrecer al menos un {reliability} de fiabilidad de entrega. {route} Prefiere la alternativa viable más ecológica y usa el precio como segundo criterio. Gestiona la solicitud con las herramientas.",
        "Encuentra una forma válida de suministrar {quantity} {unit} de {material_id} en {destination} antes del {required_date}. El presupuesto completo es de EUR {budget} y el mínimo de fiabilidad de entrega es {reliability}. {route} Tras cumplir las condiciones obligatorias, optimiza primero el carbono y luego el coste. Devuelve la decisión mediante una herramienta final.",
        "Estamos planificando un envío de {material_id}: se requieren {quantity} {unit} en {destination} para el {required_date}. Descarta cualquier opción que supere EUR {budget} en total o quede por debajo del {reliability} de fiabilidad de entrega. {route} Entre las opciones restantes, importan más las emisiones que el precio. Completa la selección con las herramientas.",
        "Consigue {quantity} {unit} de {material_id} para {destination}. Fecha límite: {required_date}; presupuesto total máximo: EUR {budget}; fiabilidad mínima de entrega: {reliability}. {route} Elige primero por menor carbono y después por menor coste, respetando esos límites, y presenta el resultado con las herramientas.",
        "¿Puedes comparar las opciones de compra para {quantity} {unit} de {material_id}, que deben estar en {destination} antes del {required_date}? No podemos gastar más de EUR {budget} en total ni aceptar menos del {reliability} de fiabilidad de entrega. {route} La sostenibilidad es la preferencia principal y el coste la segunda. Finaliza el proceso usando las herramientas.",
    ),
    "German": (
        "Wir benötigen {quantity} {unit} {material_id} bis zum {required_date} in {destination}. Die Gesamtkosten dürfen EUR {budget} nicht überschreiten und die Lieferzuverlässigkeit muss mindestens {reliability} betragen. {route} Bevorzuge zuerst geringere CO₂-Emissionen und danach niedrigere Kosten. Nutze die Werkzeuge und schließe mit einem Beschaffungsplan oder einer Meldung über keine geeignete Option ab.",
        "Beschaffe bitte {quantity} {unit} {material_id} für {destination} mit Ankunft spätestens am {required_date}. Der gesamte Auftrag muss unter EUR {budget} bleiben und eine Lieferzuverlässigkeit von mindestens {reliability} haben. {route} Bewerte bei geeigneten Optionen zuerst den CO₂-Ausstoß und dann den Preis. Führe die Aufgabe mit den Werkzeugen aus.",
        "Kannst du {material_id} beschaffen? Benötigt werden {quantity} {unit} in {destination} bis {required_date}, bei Gesamtkosten von höchstens EUR {budget} und mindestens {reliability} Lieferzuverlässigkeit. {route} Minimiere zuerst Emissionen und danach Kosten und reiche das Ergebnis über die Werkzeuge ein.",
        "Beschaffungsanfrage: {material_id}, {quantity} {unit}, Zielort {destination}, Frist {required_date}. Verbindliche Grenzen sind EUR {budget} Gesamtkosten und mindestens {reliability} Lieferzuverlässigkeit. {route} Unter den gültigen Optionen sind weniger CO₂ und anschließend geringere Kosten zu bevorzugen. Löse die Anfrage mit den verfügbaren Werkzeugen.",
        "Für die Lieferung nach {destination} bis {required_date} werden {quantity} {unit} {material_id} benötigt. Überschreite EUR {budget} nicht und verlange mindestens {reliability} Lieferzuverlässigkeit. {route} CO₂-Reduktion hat Vorrang vor dem Preis. Beende die Aufgabe mit dem passenden Abschlusswerkzeug.",
        "Ich brauche Unterstützung bei einer Bestellung über {quantity} {unit} {material_id}. Sie muss spätestens am {required_date} in {destination} eintreffen, insgesamt unter EUR {budget} kosten und mindestens {reliability} Lieferzuverlässigkeit bieten. {route} Bevorzuge die umweltfreundlichste geeignete Wahl und nutze den Preis als zweites Kriterium. Bearbeite die Anfrage mit den Werkzeugen.",
        "Finde eine zulässige Möglichkeit, {quantity} {unit} {material_id} bis {required_date} nach {destination} zu liefern. Das Gesamtbudget beträgt EUR {budget}, die Mindestanforderung an die Lieferzuverlässigkeit {reliability}. {route} Nach Erfüllung aller Pflichtbedingungen optimiere zuerst CO₂ und dann Kosten. Gib die Entscheidung über ein Abschlusswerkzeug zurück.",
        "Wir planen eine Lieferung von {material_id}: {quantity} {unit} werden bis {required_date} in {destination} benötigt. Schließe alles aus, was insgesamt mehr als EUR {budget} kostet oder unter {reliability} Lieferzuverlässigkeit liegt. {route} Bei den übrigen Optionen sind geringere Emissionen wichtiger als ein niedrigerer Preis. Schließe die Auswahl mit den Werkzeugen ab.",
        "Beschaffe {quantity} {unit} {material_id} für {destination}. Frist: {required_date}; maximales Gesamtbudget: EUR {budget}; minimale Lieferzuverlässigkeit: {reliability}. {route} Wähle innerhalb dieser Grenzen zuerst nach dem geringsten CO₂-Ausstoß und danach nach den geringsten Kosten und reiche das Ergebnis über die Werkzeuge ein.",
        "Kannst du die Beschaffungsoptionen für {quantity} {unit} {material_id} vergleichen, die bis {required_date} in {destination} sein müssen? Insgesamt dürfen wir nicht mehr als EUR {budget} ausgeben und keine Lieferzuverlässigkeit unter {reliability} akzeptieren. {route} Nachhaltigkeit ist die wichtigste Präferenz, gefolgt von den Kosten. Beende den Ablauf mit den Werkzeugen.",
    ),
    "French": (
        "Nous avons besoin de {quantity} {unit} de {material_id} livrés à {destination} avant le {required_date}. Le coût total ne doit pas dépasser EUR {budget} et la fiabilité de livraison doit atteindre au moins {reliability}. {route} Privilégie d'abord la réduction des émissions de carbone, puis le coût. Utilise les outils et termine par un plan ou un constat d'absence d'option viable.",
        "Merci d'organiser l'achat de {quantity} {unit} de {material_id} pour {destination}, avec une arrivée au plus tard le {required_date}. La commande complète doit rester sous EUR {budget} et garantir au moins {reliability} de fiabilité de livraison. {route} Parmi les choix viables, évalue d'abord l'impact carbone, puis le prix. Effectue la tâche avec les outils.",
        "Peux-tu trouver {material_id} ? Il faut {quantity} {unit} à {destination} pour le {required_date}, avec un budget total maximal de EUR {budget} et une fiabilité de livraison minimale de {reliability}. {route} Réduis d'abord les émissions, puis le coût, et soumets le résultat avec les outils.",
        "Demande d'achat : {material_id}, {quantity} {unit}, destination {destination}, échéance {required_date}. Les limites obligatoires sont EUR {budget} au total et {reliability} de fiabilité minimale de livraison. {route} Parmi les options valides, favorise un carbone plus faible, puis un coût inférieur. Résous la demande avec les outils disponibles.",
        "Pour une livraison à {destination} avant le {required_date}, procure {quantity} {unit} de {material_id}. Ne dépasse pas EUR {budget} et exige une fiabilité de livraison d'au moins {reliability}. {route} La réduction du carbone est prioritaire et le prix secondaire. Termine avec l'outil final approprié.",
        "J'ai besoin d'aide pour passer une commande de {quantity} {unit} de {material_id}. Elle doit arriver à {destination} au plus tard le {required_date}, coûter moins de EUR {budget} au total et offrir au moins {reliability} de fiabilité de livraison. {route} Préfère le choix viable le plus écologique et utilise le prix comme second critère. Traite la demande avec les outils.",
        "Trouve une solution conforme pour livrer {quantity} {unit} de {material_id} à {destination} avant le {required_date}. Le budget global est de EUR {budget} et le seuil de fiabilité de livraison est {reliability}. {route} Une fois les contraintes obligatoires respectées, optimise d'abord le carbone, puis le coût. Renvoie la décision avec un outil final.",
        "Nous préparons une expédition de {material_id} : {quantity} {unit} sont nécessaires à {destination} pour le {required_date}. Écarte toute option dépassant EUR {budget} au total ou offrant moins de {reliability} de fiabilité de livraison. {route} Parmi les choix restants, les émissions comptent davantage que le prix. Termine la sélection avec les outils.",
        "Procure {quantity} {unit} de {material_id} pour {destination}. Échéance : {required_date} ; budget total maximal : EUR {budget} ; fiabilité minimale de livraison : {reliability}. {route} Dans ces limites, choisis d'abord le carbone le plus faible, puis le coût le plus bas, et soumets le résultat avec les outils.",
        "Peux-tu comparer les solutions d'achat pour {quantity} {unit} de {material_id}, attendues à {destination} avant le {required_date} ? Nous ne pouvons pas dépasser EUR {budget} au total ni accepter une fiabilité de livraison inférieure à {reliability}. {route} La durabilité est la préférence principale, suivie du coût. Termine le processus avec les outils.",
    ),
}


def format_prompt(scenario, language="English", template_index=0):
    """Render one scenario without changing its executable semantics."""
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    templates = PROMPT_TEMPLATES[language]
    if not 0 <= template_index < len(templates):
        raise ValueError(f"template_index must be between 0 and {len(templates) - 1}")
    route = ROUTES[language][scenario["task_type"]].format(**scenario)
    return templates[template_index].format(
        **scenario,
        budget=f"{scenario['maximum_total_cost']:.0f}",
        reliability=f"{scenario['minimum_delivery_reliability']:.0%}",
        route=route,
    )


def prompt_variants(
    scenario,
    languages=LANGUAGES,
    templates_per_language=TEMPLATES_PER_LANGUAGE,
):
    """Yield prompt variants that retain the semantic scenario identifier."""
    if not 1 <= templates_per_language <= TEMPLATES_PER_LANGUAGE:
        raise ValueError(
            f"templates_per_language must be between 1 and {TEMPLATES_PER_LANGUAGE}"
        )
    for language in languages:
        for template_index in range(templates_per_language):
            variant = deepcopy(scenario)
            variant["language"] = language
            variant["prompt_variant"] = f"{language.lower()}_{template_index + 1}"
            variant["user_request"] = format_prompt(
                variant, language, template_index
            )
            yield variant
